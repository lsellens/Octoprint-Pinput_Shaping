#include "adxl345.h"
#include "interface.h"
#include "interface_i2c.h"
#include "interface_spi.h"
#include "interface_usb.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define CODE_VERSION "0.1"

volatile sig_atomic_t keepRunning = 1;
struct termios orig_term;
FILE *f = NULL;
int term_modified = 0;

void intHandler(int dummy) { keepRunning = 0; }

void printUsage() {
  printf("adxl345 (version %s)\n"
         "Usage: adxl345 [OPTION]...\n"
         "  -s, --save FILE     Save data to specified FILE\n"
         "  -f, --freq FREQ     Sampling rate in Hz (default: %d, max: %d)\n"
         "  -t, --time SECONDS  Stop after SECONDS seconds\n"
         "  -i, --interface INT Use interface: spi (default), i2c, usb\n",
         CODE_VERSION, freqDefault, freqMax);
}

double time_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

void cleanup(void) {
  interface_close();
  if (term_modified)
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_term);
  if (f)
    fclose(f);
}

int main(int argc, const char *const argv[]) {
  atexit(cleanup);
  signal(SIGINT, intHandler);
  setvbuf(stdout, NULL, _IOFBF, 4096);

  int bSave = 0;
  int duration = 0;
  char vSave[256] = "";
  double vFreq = freqDefault;
  InterfaceType iface = INTERFACE_SPI;

  if (tcgetattr(STDIN_FILENO, &orig_term) == 0) {
    struct termios new_term = orig_term;
    new_term.c_lflag &= ~(ECHO);
    if (tcsetattr(STDIN_FILENO, TCSANOW, &new_term) == 0)
      term_modified = 1;
  }

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-s") || !strcmp(argv[i], "--save")) {
      bSave = 1;
      if (++i < argc)
        strcpy(vSave, argv[i]);
      else {
        fprintf(stderr, "Error: Missing argument for -s\n");
        return 1;
      }
    } else if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--freq")) {
      if (++i < argc) {
        vFreq = atof(argv[i]);
        if (vFreq < 1 || vFreq > freqMax) {
          fprintf(stderr, "Error: Invalid freq. Must be 1-%d\n", freqMax);
          return 1;
        }
      } else {
        fprintf(stderr, "Error: Missing argument for -f\n");
        return 1;
      }
    } else if (!strcmp(argv[i], "-t") || !strcmp(argv[i], "--time")) {
      if (++i < argc)
        duration = atoi(argv[i]);
      else {
        fprintf(stderr, "Error: Missing argument for -t\n");
        return 1;
      }
    } else if (!strcmp(argv[i], "-i") || !strcmp(argv[i], "--interface")) {
      if (++i < argc) {
        if (!strcmp(argv[i], "spi"))
          iface = INTERFACE_SPI;
        else if (!strcmp(argv[i], "i2c"))
          iface = INTERFACE_I2C;
        else if (!strcmp(argv[i], "usb"))
          iface = INTERFACE_USB;
        else {
          fprintf(stderr, "Error: Invalid interface '%s'\n", argv[i]);
          return 1;
        }
      } else {
        fprintf(stderr, "Error: Missing arg for -i\n");
        return 1;
      }
    } else {
      fprintf(stderr, "Error: Unknown option '%s'\n", argv[i]);
      printUsage();
      return 1;
    }
  }

  if (interface_open(iface) != 0) {
    fprintf(stderr, "Error: Failed to open interface\n");
    return 1;
  }

  adxl345_check_devid();
  current_interface = iface;
  adxl345_init();

  if (bSave) {
    f = fopen(vSave, "w");
    if (!f) {
      fprintf(stderr, "Error opening file\n");
      return 1;
    }
    fprintf(f, "time,x,y,z\n");
  }

  printf("Press Ctrl+C to stop\n");
  fflush(stdout);
  double delay = 1.0 / vFreq, tStart = time_time();
  int totalSamples = 0;

  struct timespec interval = {(time_t)delay, (delay - (time_t)delay) * 1e9};
  struct timespec next;
  clock_gettime(CLOCK_MONOTONIC, &next);

  while (keepRunning) {
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
    if (duration > 0 && (time_time() - tStart) >= duration)
      break;

    int16_t x, y, z;
    adxl345_read_xyz(&x, &y, &z);

    double t = time_time() - tStart;
    if (bSave)
      fprintf(f, "%.5f,%.5f,%.5f,%.5f\n", t, x * accConversion,
              y * accConversion, z * accConversion);
    else
      printf("time = %.3f, x = %.3f, y = %.3f, z = %.3f\n", t,
             x * accConversion, y * accConversion, z * accConversion);

    totalSamples++;
    next.tv_nsec += interval.tv_nsec;
    next.tv_sec += interval.tv_sec;
    if (next.tv_nsec >= 1000000000) {
      next.tv_nsec -= 1000000000;
      next.tv_sec++;
    }
  }

  if (bSave && f) {
    printf("Saved %d samples in %.2f seconds (%.1f Hz) to %s\n", totalSamples,
           time_time() - tStart, totalSamples / (time_time() - tStart), vSave);
  } else {
    printf("Captured %d samples in %.2f seconds (%.1f Hz)\n", totalSamples,
           time_time() - tStart, totalSamples / (time_time() - tStart));
  }
  return 0;
}
