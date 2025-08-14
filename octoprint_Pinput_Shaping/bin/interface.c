#include "interface.h"
#include "adxl345.h"
#include "interface_i2c.h"
#include "interface_spi.h"
#include "interface_usb.h"

InterfaceType current_interface = INTERFACE_SPI;

int interface_open(InterfaceType iface) {
  current_interface = iface;

  switch (iface) {
  case INTERFACE_SPI:
    return spi_open("/dev/spidev0.0", speedSPI);
  case INTERFACE_I2C:
    return i2c_open("/dev/i2c-1", I2C_ADDR);
  case INTERFACE_USB:
    return usb_open("/dev/ttyUSB0");
  default:
    return -1;
  }
}

void interface_close(void) {
  switch (current_interface) {
  case INTERFACE_SPI:
    spi_close();
    break;
  case INTERFACE_I2C:
    i2c_close();
    break;
  case INTERFACE_USB:
    usb_close();
    break;
  }
}

int interface_transfer(uint8_t *tx, uint8_t *rx, size_t len) {
  switch (current_interface) {
  case INTERFACE_SPI:
    // SPI just sends exactly len bytes
    return spi_transfer(tx, rx, len);

  case INTERFACE_I2C:
    if (len == 2) {
      // Case: device ID read (tx[0] = reg, tx[1] = dummy)
      return i2c_transfer(tx, 1, rx, 1);
    } else if (len == 7) {
      // Case: XYZ read (tx[0] = reg, rest are dummy)
      return i2c_transfer(tx, 1, rx, 6);
    } else {
      // Generic write-then-read
      return i2c_transfer(tx, 1, rx, len - 1);
    }

  case INTERFACE_USB:
    return usb_transfer(tx, len, rx, len); // still stub

  default:
    return -1;
  }
}

int interface_write(uint8_t reg, uint8_t val) {
  uint8_t buf[2] = {reg, val};

  switch (current_interface) {
  case INTERFACE_SPI:
    return spi_transfer(buf, NULL, 2);
  case INTERFACE_I2C:
    return i2c_transfer(buf, 2, NULL, 0);
  case INTERFACE_USB:
    return usb_transfer(buf, 2, NULL, 0); // still stub
  default:
    return -1;
  }
}
