import sys
sys.path.append("./")

import os
import numpy as np

from camera.camera import Camera
from serial_com.comms import SerialComm

class Main(Camera, SerialComm):
    # Camera Setup Options
    camera_config = {
        "video_format": ".avi",
        "save_folder": "", # where you want to save the file
        "file_name": "python_camera",
        "n_cameras": 2,
        "timeout": 100, 
        "outputdict":{
            '-vcodec': 'libx264',
            '-crf': '0',
            '-preset': 'slow',
            '-pix_fmt': 'yuvj444p',
            '-framerate': '30'
        }
    }

    # Serial Comms (Arduino) options
    com_port = "COM5"
    baudrate = 115200

    def __init__(self):
        Camera.__init__(self)
        SerialComm.__init__(self)
        


if __name__ == "__main__":
    m = Main()
    m.start_cameras()
    m.stream_videos()

    # m.conneFLUSBVGA-1.1.323.0ct_serial()