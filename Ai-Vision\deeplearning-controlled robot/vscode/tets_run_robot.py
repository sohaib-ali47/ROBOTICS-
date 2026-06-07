import serial
import time

arduino = serial.Serial('COM7', 9600, timeout=1)
time.sleep(2)

while True:
    arduino.write(b'F')
    print("sent F")
    time.sleep(2)

    arduino.write(b'S')
    print("sent S")
    time.sleep(2)