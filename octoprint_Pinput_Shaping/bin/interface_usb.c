#include "interface_usb.h"
#include <stdio.h>
#include <unistd.h>

static int usb_fd = -1;

int usb_open(const char *device) {
  (void)device;
  fprintf(stderr, "USB interface not implemented yet.\n");
  return -1;
}

void usb_close(void) {
  if (usb_fd >= 0) {
    close(usb_fd);
    usb_fd = -1;
  }
}

int usb_transfer(uint8_t *tx, int tx_len, uint8_t *rx, int rx_len) {
  (void)tx;
  (void)tx_len;
  (void)rx;
  (void)rx_len;
  fprintf(stderr, "USB transfer not implemented yet.\n");
  return -1;
}
