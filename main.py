import sys

sys.path.append("./")

import os
import numpy as np
from threading import Thread
import time

from camera.camera import Camera
from serial_com.comms import SerialComm
from utils.file_io_utils import *

from analysis.analysis import Analyzer
from forceplate_config import Config


class Main(Camera, SerialComm, Analyzer, Config):
    _tone_command = 0  # used to write to csv when tone command starts and ends
    _door_command = 0  # used to write to csv when door command starts and ends

    def __init__(self):
        Config.__init__(self)  # load the config paramaters
        Camera.__init__(self)
        SerialComm.__init__(self)

    def setup_experiment_files(self):
        # Takes care of creating a folder to keep the files of this experiment
        # Checks if files exists already in that folder
        # Checks if we are overwriting anything
        # Creates a csv file to store arduino sensors data

        # Check if exp folder exists and if it's empty
        check_create_folder(self.experiment_folder)
        if not check_folder_empty(self.experiment_folder):
            print(
                "\n\n!!! experiment folder is not empty, might risk overwriting stuff !!!\n\n"
            )

        # Create files for videos
        if self.save_to_video:
            self.video_files_names = [
                os.path.join(
                    self.experiment_folder,
                    self.experiment_name
                    + "_cam{}{}".format(i, self.camera_config["video_format"]),
                )
                for i in np.arange(self.camera_config["n_cameras"])
            ]

            # Check if they exist already
            for vid in self.video_files_names:
                if check_file_exists(vid) and not self.overwrite_files:
                    raise FileExistsError("Cannot overwrite video file: ", vid)

        # Creat csv file for arduino saving
        self.arduino_inputs_file = os.path.join(
            self.experiment_folder, self.experiment_name + "_analoginputs.csv"
        )
        if check_file_exists(self.arduino_inputs_file) and not self.overwrite_files:
            raise FileExistsError(
                "Cannot overwrite analog inputs file: ", self.arduino_inputs_file
            )
        create_csv_file(
            self.arduino_inputs_file, self.arduino_config["arduino_csv_headers"]
        )

    def terminate_experiment(self):
        """
            This function gets called when the user interrupts the execution of the experiments.
            It takes care of printing a summary, plotting stuff, closing the ffmpeg writers etc etc
        """
        exp_duration = round(time.time() - self.exp_start_time / 1000, 2)
        print(
            """\n\n\nTerminating experiment. Acquired {} frames in {}s.
                {}s / {}fps -> {} frames, 
                {} frames acquired, actual framerate: {}""".format(
                self.frame_count,
                exp_duration,
                exp_duration,
                self.acquisition_framerate,
                int(exp_duration * self.acquisition_framerate),
                self.frame_count,
                round(self.frame_count / exp_duration, 2),
            )
        )

        # Close pylon windows and ffmpeg writers
        self.close_pylon_windows()
        self.close_ffmpeg_writers()

    def start_experiment(self):
        self.parallel_processes = []  # store all the parallel processes

        # Start cameras and set them up`
        self.start_cameras()

        # Start the arduino connection
        self.connect_firmata()
        self.setup_pins()

        # Start live plotting
        if self.live_plotting:
            self.initialise_sensors_plot()

        # Start streaming videos
        self.exp_start_time = (
            time.time() * 1000
        )  #  experiment starting time in milliseconds

        try:
            self.stream_videos()  # <- t
        except Exception as e:
            print("Acquisition terminted with error: ", e)
            self.terminate_experiment()


if __name__ == "__main__":
    m = Main()
    m.setup_experiment_files()
    m.start_experiment()
