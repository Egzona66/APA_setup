import sys
sys.path.append("./")

import os
import numpy as np
import matplotlib.pyplot as plt
import time

from camera.camera import Camera
from serial_com.comms import SerialComm


class SerialTest(SerialComm):
    acquisition_framerate = 100

    # Serial Comms (Arduino) options
    com_port = "COM3"
    baudrate = 115200

    def __init__(self):
        SerialComm.__init__(self)

        # self.connect_serial()

    def test_read(self, acquire_for = 100):
        data = np.zeros((acquire_for, 5))

        delta_t = []
        start = time.time()
        acquired = 0
        while True:
            read = self.read_serial()
            if read is not None: 
                data[acquired, :] = read
                acquired += 1
                print(acquired)
                now = time.time()
                delta_t.append((now-start)*1000)
                start = now

                if acquired == acquire_for: break

        f, ax = plt.subplots()
        ax.plot(delta_t)
        plt.show()

    def test_firmata(self):
        self.connect_firmata()

        self.camera_triggers()

        # pin = self.arduino.get_pin("d:10:i")
        # print(pin.read())





if __name__ == "__main__":
    t = SerialTest()
    t.test_firmata()
    # t.get_available_ports()