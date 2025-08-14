#include "interface_spi.h"
#include <fcntl.h>
#include <linux/spi/spidev.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <unistd.h>

static int spi_fd = -1;
const int speedSPI = 2000000;

int spi_open(const char *device, uint32_t speed) {
  uint8_t mode = SPI_MODE_3;
  uint8_t bits = 8;

  spi_fd = open(device, O_RDWR);
  if (spi_fd < 0)
    return -1;

  if (ioctl(spi_fd, SPI_IOC_WR_MODE, &mode) == -1)
    return -1;
  if (ioctl(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &bits) == -1)
    return -1;
  if (ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) == -1)
    return -1;

  return 0;
}

void spi_close(void) {
  if (spi_fd >= 0) {
    close(spi_fd);
    spi_fd = -1;
  }
}

int spi_transfer(uint8_t *tx, uint8_t *rx, int len) {
  struct spi_ioc_transfer tr = {
      .tx_buf = (unsigned long)tx,
      .rx_buf = (unsigned long)rx,
      .len = len,
      .speed_hz = speedSPI,
      .bits_per_word = 8,
      .delay_usecs = 0,
  };

  return ioctl(spi_fd, SPI_IOC_MESSAGE(1), &tr);
}
