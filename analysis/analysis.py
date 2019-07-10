import sys
sys.path.append("./")

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import warnings

from utils.video_utils import Editor as VideoUtils
from forceplate_config import Config


from utils.file_io_utils import *
from utils.analysis_utils import *


class Analyzer(Config):
    def __init__(self, posthoc=False):
        plt.ioff()  # make sure plt is not in interactive mode

        # If posthoc is true, load data from a file specified in the Config class
        # otherwise load the data from the experiment just terminated (specified in Main>Config)
        if not posthoc:
            self.data = load_csv_file(self.arduino_inputs_file)
        else:
            Config.__init__(self)
            arduino_file = os.path.join(self.analysis_config["data_folder"], self.analysis_config["experiment_name"]+"_analoginputs.csv")
            
            if not check_file_exists(arduino_file): raise FileExistsError("analysis file specified in config does not exist: {}".format(arduino_file))
            
            self.data = load_csv_file(arduino_file)

        # Get video utils functions
        self.video_utils = VideoUtils()

        # Keep track of opened figures
        self.figures = {}


    """
            #############   DATA QUALITY CONTROL FUNCTIONS
    """

    def check_number_frames(self):           
        # run this at the end of an experiment to check that the number of frames in the videos and csv files is correct

        if not self.save_to_video: return # we didnt save any video

        # number of rows in CSV file
        csv_frames = len(self.data) # length of the loaded data pd.DataFrame

        # Number of frames in the two videos
        nframes, width, height, fps = [], [], [], []
        for video in self.video_files_names:
            nf, w, h, r = self.video_utils.get_video_params(video)
            nframes.append(nf)
            width.append(w)
            height.append(h)
            fps.append(r)

        # Check if all the frame numbers are correct!
        s1 = """
Frames recorded: {}
Frames in csv:   {}
Frames in vids:  {}, {}
            
            """.format(self.frame_count, csv_frames, nframes[0], nframes[1])

        s2 = """
Acquisition framerate: {}
Videos framerates:      {}, {}
            """.format(self.acquisition_framerate, fps[0], fps[1])



        if self.frame_count == csv_frames == nframes[0] == nframes[1]:
            # all good
            print("Number of frames is correct everywhere")
        else:
            # warnings.warn("The number of frames is not the same everywhere!!!")
            print(s1)

        # Check if the fps of the videos saved is the same as what we would've liked
        if self.acquisition_framerate == fps[0] == fps[1]:
            # all good
            print("Videos where saved at: ", self.acquisition_framerate)
        else:
            # warnings.warn(s2)
            pass

        # Write outcome to log file
        with open(os.path.join(self.experiment_folder, self.experiment_name+"_log.txt"), "w") as log:
            log.writelines([s1, s2])



    """
            #############   PLOTTING FUNCTIONS
    """

    def plot_frame_delays(self):
        f, axarr = plt.subplots(figsize=(12, 10), ncols=2)

        try:
            dt = np.diff(self.data.elapsed.values)
        except:
            print("could not print frame delays")
            return
        mean_dt, std_dt = round(np.mean(dt), 2), round(np.std(dt), 2)

        axarr[0].plot(dt, label="frames - elapsed")
        axarr[0].axhline(mean_dt, color="r", lw=2, label="mean")
        axarr[0].legend()
        axarr[0].set(xlabel="frame number", ylabel="delta t ms", title="Frame ITI - {}ms +- {}ms".format(mean_dt, std_dt))

        camera_dt = np.diff(self.data.camera_timestamp.values)

        axarr[1].plot(camera_dt, label="frames - camera time")
        axarr[1].axhline(np.mean(camera_dt), color="r", lw=2, label="mean")
        axarr[1].legend()
        axarr[1].set(xlabel="frame number", ylabel="delta t", title="Frame ITI")
        self.figures[self.experiment_name+"_frames_deltaT.png"] = f

    def plot_sensors_traces(self):
        f, axarr = plt.subplots(nrows=2, sharex=True, figsize=(12, 10))

        try:
            normalized = {ch: self.data[ch].values / np.nanmean(self.data[ch].values) for ch in self.arduino_config["sensors"]}
        except: # cant normalise if mean of channel is zero
            normalized = None
    
        # plot raw and normalized
        for ch, color in self.analysis_config["plot_colors"].items():
            axarr[0].plot(self.data[ch], color=color, label=ch)
            if normalized is not None: axarr[1].plot(normalized[ch], color=color, label=ch)

        for ax in axarr: ax.legend()
        axarr[0].set(title="Raw Force Sensor Data", xlabel="frames", ylabel="Volts")
        axarr[1].set(title="Normalized Force Sensor Data", xlabel="frames", ylabel="Volts")

        self.figures[self.experiment_name+"_sensors_traces.png"] = f



    def plot_sensors_traces_fancy(self):
        f, ax = plt.subplots(figsize=(12, 10))
        normalized = normalize_channel_data(self.data, self.arduino_config["sensors"])

        for ch, color in self.analysis_config["plot_colors"].items():
            # channel_data = self.data[ch].values
            channel_data = normalized[ch]

            x = np.arange(0, len(channel_data))
            ax.fill_between(x, 0, channel_data, color=color, label=ch, alpha=.3)
        
        ax.legend()
        ax.set(title="Raw Force Sensor Data", xlabel="frames", ylabel="Volts", facecolor=[.5, .5, .5])

        self.figures[self.experiment_name+"_sensors_traces_fancy.png"] = f


    def plot_sensors_traces_fancy_separated(self):
        f, axarr = plt.subplots(figsize=(12, 10), nrows=4, sharex=True, sharey=True)

        normalized = normalize_channel_data(self.data, self.arduino_config["sensors"])

        for i, (ch, color) in enumerate(self.analysis_config["plot_colors"].items()):
            # channel_data = self.data[ch].values
            channel_data = normalized[ch]

            x = np.arange(0, len(channel_data))
            axarr[i].fill_between(x, 0, channel_data, color=color, label=ch, alpha=.3)
        
            axarr[i].legend()
            axarr[i].set(title="Raw Force Sensor Data", xlabel="frames", ylabel="Volts", facecolor=[.5, .5, .5], ylim=[0,1])

        self.figures[self.experiment_name+"_sensors_traces_fancy.png"] = f



    def save_figs(self):
        for fname, f in self.figures.items():
            print("Saving: ", os.path.join(self.analysis_config["data_folder"], fname))
            f.savefig(os.path.join(self.experiment_folder, fname))

    def show(self):
        try: plt.show()
        except KeyboardInterrupt: pass


if __name__ == "__main__":
    analyzer = Analyzer(posthoc=True)
    analyzer.plot_frame_delays()
    analyzer.plot_sensors_traces_fancy()
    analyzer.plot_sensors_traces_fancy_separated()
    plt.show()