#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/select.h>
#include <time.h>
#include <signal.h>

#define BUF_SIZE 4096
#define FLUSH_INTERVAL 1000
#define MAX_LINE_LENGTH 256
#define COLD_START_SKIP 1

struct termios oldt;

void restore_terminal() {
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
}

int set_serial_port(int fd, int speed) {
    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) return -1;

    cfsetospeed(&tty, speed);
    cfsetispeed(&tty, speed);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 1;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    return tcsetattr(fd, TCSANOW, &tty);
}

int main(int argc, char **argv) {
    const char *port = "/dev/ttyACM0";
    const char *outfile = NULL;
    int capture_time = 0;
    int freq = 250;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p")) port = argv[++i];
        else if (!strcmp(argv[i], "-s")) outfile = argv[++i];
        else if (!strcmp(argv[i], "-t")) capture_time = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-f")) freq = atoi(argv[++i]);
        else {
            fprintf(stderr, "Usage: %s [-p port] [-s file.csv] [-t seconds] [-f freq]\n", argv[0]);
            exit(1);
        }
    }

    int fd = open(port, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) { perror("open serial port"); exit(2); }

    if (set_serial_port(fd, B2000000) < 0) { perror("serial config"); exit(2); }

    tcflush(fd, TCIFLUSH);

    char cmd[32];
    snprintf(cmd, sizeof(cmd), "F=%d\n", freq);
    ssize_t written = write(fd, cmd, strlen(cmd));
    if (written < 0) { perror("write"); exit(2); }

    FILE *csv = NULL;
    if (outfile) {
        csv = fopen(outfile, "w");
        if (!csv) { perror("fopen"); exit(3); }
        fprintf(csv, "time,x,y,z\n");
    }

    printf("Press Q to stop\n");
    fflush(stdout);

    tcgetattr(STDIN_FILENO, &oldt);
    struct termios newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    atexit(restore_terminal);

    char read_buf[BUF_SIZE];
    char line_buffer[MAX_LINE_LENGTH];
    int line_buffer_pos = 0;
    int samples = 0;
    int skip = COLD_START_SKIP;
    double t, x, y, z;

    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int fdmax = fd > STDIN_FILENO ? fd : STDIN_FILENO;

    while (1) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(STDIN_FILENO, &rfds);
        FD_SET(fd, &rfds);
        struct timeval tv = {0, 5000};

        int r = select(fdmax + 1, &rfds, NULL, NULL, &tv);
        if (r < 0 && errno != EINTR) { perror("select"); break; }

        if (FD_ISSET(STDIN_FILENO, &rfds)) {
            char c = getchar();
            if (c == 'q' || c == 'Q') break;
        }

        if (FD_ISSET(fd, &rfds)) {
            int bytes_read = read(fd, read_buf, BUF_SIZE);
            if (bytes_read < 0) { perror("read serial port"); break; }

            for (int i = 0; i < bytes_read; i++) {
                char c = read_buf[i];
                if (c == '\n') {
                    if (line_buffer_pos > 0) {
                        line_buffer[line_buffer_pos] = '\0';
                        if (sscanf(line_buffer, "%lf,%lf,%lf,%lf", &t, &x, &y, &z) == 4) {
                            if (skip > 0) {
                                skip--;
                            } else {
                                samples++;
                                if (csv) {
                                    fprintf(csv, "%.6f,%.6f,%.6f,%.6f\n", t, x, y, z);
                                    if (samples % FLUSH_INTERVAL == 0) fflush(csv);
                                } else {
                                    printf("time = %.3f, x = %.3f, y = %.3f, z = %.3f\n", t, x, y, z);
                                }
                            }
                        }
                    }
                    line_buffer_pos = 0;
                } else if (line_buffer_pos < MAX_LINE_LENGTH - 1) {
                    line_buffer[line_buffer_pos++] = c;
                } else {
                    line_buffer_pos = 0;
                }
            }
        }

        if (capture_time > 0) {
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed = now.tv_sec - start.tv_sec + (now.tv_nsec - start.tv_nsec) / 1e9;
            if (elapsed >= capture_time) break;
        }
    }

    close(fd);
    if (csv) {
        fflush(csv);
        fclose(csv);
    }

    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed = now.tv_sec - start.tv_sec + (now.tv_nsec - start.tv_nsec) / 1e9;

    if (csv)
        printf("Saved %d samples in %.2f seconds (%.1f Hz) to %s\n",
                samples, elapsed, samples / elapsed, outfile);
    else
        fprintf(stderr, "Captured %d samples in %.2f s (%.1f Hz)\n",
                samples, elapsed, samples / elapsed);
    return 0;
}
