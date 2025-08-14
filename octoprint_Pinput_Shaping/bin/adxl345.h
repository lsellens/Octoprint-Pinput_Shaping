#ifndef ADXL345_H
#define ADXL345_H

#include <stdint.h>

#define READ_BIT 0x80
#define MULTI_BIT 0x40
#define BW_RATE 0x2C
#define DATA_FORMAT 0x31
#define POWER_CTL 0x2D
#define DATAX0 0x32
#define DEVID 0x00
#define EXPECTED_DEVID 0xE5

extern const uint8_t I2C_ADDR;
extern const int freqDefault;
extern const int freqMax;
extern const double accConversion;

void adxl345_init(void);
void adxl345_check_devid(void);
void adxl345_read_xyz(int16_t *x, int16_t *y, int16_t *z);

#endif
