#ifndef INTERFACE_I2C_H
#define INTERFACE_I2C_H

#include <stdint.h>

int i2c_open(const char *device, uint8_t addr);
void i2c_close(void);
int i2c_transfer(uint8_t *tx, int tx_len, uint8_t *rx, int rx_len);

#endif
