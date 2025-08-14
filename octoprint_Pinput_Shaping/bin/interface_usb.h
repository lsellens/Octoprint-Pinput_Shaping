#ifndef INTERFACE_USB_H
#define INTERFACE_USB_H

#include <stdint.h>

int usb_open(const char *device);
void usb_close(void);
int usb_transfer(uint8_t *tx, int tx_len, uint8_t *rx, int rx_len);

#endif
