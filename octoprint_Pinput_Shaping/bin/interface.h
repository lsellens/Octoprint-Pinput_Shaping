#ifndef INTERFACE_H
#define INTERFACE_H

#include <stddef.h>
#include <stdint.h>

typedef enum { INTERFACE_SPI, INTERFACE_I2C, INTERFACE_USB } InterfaceType;

extern InterfaceType current_interface;

int interface_open(InterfaceType iface);
void interface_close(void);
int interface_transfer(uint8_t *tx, uint8_t *rx, size_t len);
int interface_write(uint8_t reg, uint8_t val);

#endif
