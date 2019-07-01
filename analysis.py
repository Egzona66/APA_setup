import sys
sys.path.append("./")

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt


from utils.file_io_utils import *


class Analyzer:
    def __init__(self):
        self.data = load_csv_file(self.arduino_inputs_file)


    def plot_frame_delays(self):
        _, ax = plt.subplots()

        ax.plot(np.diff(self.data.elapsed.values), label="single frames")
        ax.axhline(np.mean(np.diff(self.data.elapsed.values)), color="r", lw=2, label="mean")
        ax.legend()
        ax.set(xlabel="frame number", ylabel="delta t ms", title="Frame ITI")

    def plot_sensors_traces(self):
        _, axarr = plt.subplots(nrows=2)

        try:
            normalized = {ch: self.data[ch].values / np.nanmean(self.data[ch].values) for ch in self.arduino_config["sensors"]}
        except: # cant normalise if mean of channel is zero
            normalized = None
    
        # plot raw and normalized
        for ch, color in self.arduino_config["plot_colors"].items():
            axarr[0].plot(self.data[ch], color=color, label=ch)
            if normalized is not None: axarr[1].plot(normalized[ch], color=color, label=ch)

        for ax in axarr: ax.legend()
        axarr[0].set(title="Raw Force Sensor Data", xlabel="frames", ylabel="Volts")
        axarr[1].set(title="Normalized Force Sensor Data", xlabel="frames", ylabel="Volts")

    def show(self):
        plt.show()


if __name__ == "__main__":
    plot_frame_delays()