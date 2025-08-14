#include "interface_i2c.h"
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

static int i2c_fd = -1;
static uint8_t i2c_addr = 0;

int i2c_open(const char *device, uint8_t addr) {
  i2c_fd = open(device, O_RDWR);
  if (i2c_fd < 0)
    return -1;
  i2c_addr = addr;
  return 0;
}

void i2c_close(void) {
  if (i2c_fd >= 0) {
    close(i2c_fd);
    i2c_fd = -1;
  }
}

int i2c_transfer(uint8_t *tx, int tx_len, uint8_t *rx, int rx_len) {
  if (i2c_fd < 0)
    return -1;

  struct i2c_rdwr_ioctl_data ioctl_data;
  struct i2c_msg msgs[2];

  memset(msgs, 0, sizeof(msgs));
  memset(&ioctl_data, 0, sizeof(ioctl_data));

  int msg_count = 0;

  if (tx_len > 0) {
    msgs[msg_count].addr = i2c_addr;
    msgs[msg_count].flags = 0;
    msgs[msg_count].len = tx_len;
    msgs[msg_count].buf = tx;
    msg_count++;
  }

  if (rx_len > 0) {
    msgs[msg_count].addr = i2c_addr;
    msgs[msg_count].flags = I2C_M_RD;
    msgs[msg_count].len = rx_len;
    msgs[msg_count].buf = rx;
    msg_count++;
  }

  ioctl_data.msgs = msgs;
  ioctl_data.nmsgs = msg_count;

  if (ioctl(i2c_fd, I2C_RDWR, &ioctl_data) < 0)
    return -1;

  return tx_len + rx_len;
}
