import sys
sys.path.append("./")

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import warnings

from utils.video_utils import Editor as VideoUtils


from utils.file_io_utils import *


class Analyzer(Config):
    def __init__(self, posthoc=False):
        plt.ioff()  # make sure plt is not in interactive mode

        # If posthoc is true, load data from a file specified in the Config class
        # otherwise load the data from the experiment just terminated (specified in Main>Config)
        if not posthoc:
            self.data = load_csv_file(self.arduino_inputs_file)
        else:
            Config.__init__(self)
            arduino_file = os.path.join(self.analysis_config["data_folder"], self.analysis_config["experiment_name"])
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
        if self.frame_count == csv_frames == nframes[0] == nframes[1]:
            # all good
            print("Number of frames is correct everywhere")
        else:
            warnings.warn("The number of frames is not the same everywhere!!!")
            print("""
            Frames recorded: {}
            Frames in csv:   {}
            Frames in vids:  {}, {}
            
            """.format(self.frame_count, csv_frames, nframes[0], nframes[1]))

        # Check if the fps of the videos saved is the same as what we would've liked
        if self.acquisition_framerate == fps[0] == fps[1]:
            # all good
            print("Videos where saved at: ", self.acquisition_framerate)
        else:
            warnings.warn("The framerate of the saved videos is not the same as the acquisition framerate")
            print("""
            Acquisition framerate: {}
            Videos framerates:      {}, {}
            """.format(self.acquisition_framerate, fps[0], fps[1]))



    """
            #############   PLOTTING FUNCTIONS
    """

    def plot_frame_delays(self):
        f, ax = plt.subplots(figsize=(12, 10))

        dt = np.diff(self.data.elapsed.values)
        mean_dt, std_dt = np.mean(dt), np.std(dt)

        ax.plot(dt, label="single frames")
        ax.axhline(mean_dt, color="r", lw=2, label="mean")
        ax.legend()
        ax.set(xlabel="frame number", ylabel="delta t ms", title="Frame ITI - {}ms +- {}ms".format(mean_dt, std_dt))

        self.figures[self.analysis_config["experiment_name.png"]+"frames_deltaT"] = f

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

        self.figures[self.analysis_config["experiment_name"]+"sensors_traces.png"] = f

    def plot_sensors_traces_fancy(self):
        f, ax = plt.subplots(figsize=(12, 10))

        for ch, color in self.analysis_config["plot_colors"].items():
            channel_data = self.data[ch].values
            x = np.linspace(0, channel_data, num=channel_data)
            ax.fill(x, channel_data, color=color, label=ch)

        ax.set(title="Raw Force Sensor Data", xlabel="frames", ylabel="Volts")

        self.figures[self.analysis_config["experiment_name"]+"sensors_traces_fancy.png"] = f

    def save_figs(self):
        for fname, f in self.figures.items():
            f.savefig(os.path.join(self.analysis_config["experiment_name.png"], fname))

    def show(self):
        plt.show()


if __name__ == "__main__":
    analyzer = Analyzer()
    analyzer.plot_sensors_traces()