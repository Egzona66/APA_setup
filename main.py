import sys
sys.path.append("./")

import os
import numpy as np
from threading import Thread
import time

from camera.camera import Camera
from serial_com.comms import SerialComm
from utils.file_io_utils import *

from analysis import Analyzer
from live_plotting import Plotter

class Main(Camera, SerialComm, Analyzer, Plotter):
    overwrite_files = False # ! ATTENTION: this is useful for debug but could lead to overwriting experimental data

    # ? General options
    acquisition_framerate = 100  # fps of camera triggering -> NEED TO SPECIFY SLEEP TIME IN ARDUINO for frame triggering
    com_port = "COM5"  # port of the arduino running Firmata for data acquisition

    experiment_folder = "E:\\Egzona"   # ? This should be changed for everyexperiment to avoid overwriting 
    experiment_name = "test_ground"  # should be something like YYMMDD_MOUSEID, all files for an experiment will start with this name

    # Camera Setup Options
    camera_config = {
        "save_to_video": False,
        "video_format": ".mp4",
        "n_cameras": 2,
        "timeout": 100,   # frame acquisition timeout

        "live_display": True,  # show the video frames as video is acquired

        # ? Trigger mode and acquisition options
        "trigger_mode": True,  # hardware triggering
        "acquisition": {    
            "exposure": "5000",
            "frame_width": "1216",  # must be a multiple of 32
            "frame_height": "1024", # must be a multiple of 32
            "gain": "10",
            "frame_offset_y": "170",
        },

        "outputdict":{ # for ffmpeg
            '-vcodec': 'mpeg4',  # ! high fps low res
            # "-vcodec": "libx264",   # ! low fps high res
            '-crf': '0',
            '-preset': 'slow',  # TODO check this
            '-pix_fmt': 'yuvj444p',
            #"-framerate": "10000", # ! output video framerate 
            # TODO this doesnt work FPS
        },
    }

    # Arduino (FIRMATA) setup options
    arduino_config = {
        "sensors_pins": {
            # Specify the pins receiving the input from the sensors
            "fr": 0, # Currently the inputs from the force sensors go to digital pins on the arduino board
            "fl": 2,
            "hr": 4, 
            "hl": 6,
        },
        "arduino_csv_headers": ["frame_number", "elapsed", "camera_timestamp", "fr", "fl", "hr", "hl"],
        "sensors": [ "fr", "fl", "hr", "hl"],
        "plot_colors": { "fr":"m", 
                        "fl":"b", 
                        "hr":"g", 
                        "hl":"r"}
    }


    def __init__(self):
        Camera.__init__(self)
        SerialComm.__init__(self)
        Plotter.__init__(self)


    def setup_experiment_files(self):
        # Takes care of creating a folder to keep the files of this experiment
        # Checks if files exists already in that folder
        # Checks if we are overwriting anything
        # Creates a csv file to store arduino sensors data

        # Check if exp folder exists and if it's empty
        check_create_folder(self.experiment_folder)
        if not check_folder_empty(self.experiment_folder):
            print("\n\n!!! experiment folder is not empty, might risk overwriting stuff !!!\n\n")

        # Create files for videos
        if self.camera_config["save_to_video"]:
            self.video_files_names = [os.path.join(self.experiment_folder, self.experiment_name+"_cam{}_{}".format(i, self.camera_config["video_format"])) for i in np.arange(self.camera_config["n_cameras"])]

            # Check if they exist already
            for vid in self.video_files_names:
                if check_file_exists(vid) and not self.overwrite_files: raise FileExistsError("Cannot overwrite video file: ", vid)

        # Creat csv file for arduino saving
        self.arduino_inputs_file = os.path.join(self.experiment_folder, self.experiment_name + "_analoginputs.csv")
        if check_file_exists(self.arduino_inputs_file) and not self.overwrite_files: raise FileExistsError("Cannot overwrite analog inputs file: ", self.arduino_inputs_file)
        create_csv_file(self.arduino_inputs_file, self.arduino_config["arduino_csv_headers"])



    def start_experiment(self):
        self.parallel_processes = [] # store all the parallel processes

        # Start cameras and set them up
        self.start_cameras()

        # Start the arduino connection
        self.connect_firmata()
        self.setup_pins()

        # Start streaming videos
        self.exp_start_time = time.time() * 1000 #  experiment starting time in milliseconds

        try:
            self.stream_videos()
        except (KeyboardInterrupt, ValueError):
            print("\n\n\nTerminating experiment. Acquired {} frames in {}s".format(self.frame_count, time.time()-self.exp_start_time/1000))
            
            # Close pylon windows and ffmpeg writers
            self.close_pylon_windows()
            self.close_ffmpeg_writers()

            # Plot stuff
            Analyzer.__init__(self)

            self.plot_frame_delays()
            self.plot_sensors_traces()
            self.show()
            


        # ? code below is to have camera and arduino run in parallel on separate threads, not necessary for now
        """
            # set up arduino camera pulses on a separate thread
            Thread(target=self.camera_triggers).start()

            # set up cameras video streaming on a separate thrad
            Thread(target=self.stream_videos).start()
                
            try: 
                # start all threads and join them
                for t in self.parallel_processes: t.start()
                for t in self.parallel_processes: t.join()
                for t in self.parallel_processes: t.lock()

            except (KeyboardInterrupt, SystemExit):
                # print '\n! Received keyboard interrupt, quitting threads.\n'
                sys.exit()
                # for t in self.parallel_processes: t.stop()
        """

        


if __name__ == "__main__":
    m = Main()
    m.setup_experiment_files()
    m.start_experiment()

    # m.start_cameras()
    # m.stream_videos()

    # m.conneFLUSBVGA-1.1.323.0ct_serial()