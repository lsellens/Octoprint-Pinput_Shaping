CC = gcc
CFLAGS = -Wall -O2

OBJS = main.o adxl345.o interface.o interface_spi.o interface_i2c.o interface_usb.o

accelerometer: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o accelerometer
