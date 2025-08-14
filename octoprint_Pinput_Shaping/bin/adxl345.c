#include "adxl345.h"
#include "interface.h"
#include <stdio.h>
#include <stdlib.h>

const uint8_t I2C_ADDR = 0x53;
const int freqDefault = 250;
const int freqMax = 3200;
const double accConversion = 2 * 16.0 / 8192.0;

void adxl345_init(void) {
  if (interface_write(BW_RATE, 0x0F) < 0) {
    fprintf(stderr, "Error writing BW_RATE\n");
    exit(1);
  }
  if (interface_write(DATA_FORMAT, 0x0B) < 0) {
    fprintf(stderr, "Error writing DATA_FORMAT\n");
    exit(1);
  }
  if (interface_write(POWER_CTL, 0x08) < 0) {
    fprintf(stderr, "Error writing POWER_CTL\n");
    exit(1);
  }
}

void adxl345_check_devid(void) {
  uint8_t tx[2] = {DEVID | READ_BIT, 0x00};
  uint8_t rx[2] = {0};

  if (interface_transfer(tx, rx, 2) < 0) {
    fprintf(stderr, "Error: Failed to read device ID\n");
    exit(1);
  }

  uint8_t devid = (current_interface == INTERFACE_SPI) ? rx[1] : rx[0];

  if (devid != EXPECTED_DEVID) {
    fprintf(stderr, "Error: Unexpected device ID 0x%02X (expected 0x%02X)\n",
            devid, EXPECTED_DEVID);
    exit(1);
  }
}

void adxl345_read_xyz(int16_t *x, int16_t *y, int16_t *z) {
  uint8_t tx[7] = {DATAX0 | READ_BIT | MULTI_BIT};
  uint8_t rx[7] = {0};

  if (interface_transfer(tx, rx, 7) < 0) {
    fprintf(stderr, "Error: Failed to read XYZ data\n");
    exit(1);
  }

  int offset = (current_interface == INTERFACE_SPI) ? 1 : 0;

  *x = (rx[offset + 1] << 8) | rx[offset + 0];
  *y = (rx[offset + 3] << 8) | rx[offset + 2];
  *z = (rx[offset + 5] << 8) | rx[offset + 4];
}
