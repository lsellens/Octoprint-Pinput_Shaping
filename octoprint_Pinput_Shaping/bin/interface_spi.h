#ifndef INTERFACE_SPI_H
#define INTERFACE_SPI_H

#include <stdint.h>

int spi_open(const char *device, uint32_t speed);
void spi_close(void);
int spi_transfer(uint8_t *tx, uint8_t *rx, int len);
extern const int speedSPI;

#endif
