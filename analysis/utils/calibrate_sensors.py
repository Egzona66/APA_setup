import sys
sys.path.append("./")

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import resample


from fcutils.maths.filtering import line_smoother
from fcutils.video.video_editing import Editor as VideoUtils
from fcutils.file_io.utils import *
from fcutils.file_io.io import load_csv_file
from fcutils.plotting.utils import *
from fcutils.plotting.colors import red, blue, green, pink, magenta, white


analysis_config = {
    "plot_colors": { "fr":magenta, 
                     "fl":blue, 
                     "hr":red, 
                     "hl":green},}

    # # * for composite video
    # # ? run video_analysis.py
    # "start_clip_time_s": None, # ? Create clips start at this point, in SECONDS
    # "start_clip_time_frame": 9799, # ? Create clips start at this point, in FRAMES
    # "clip_n_frames": 180 , # duration of the clip in frames
    # "clip_name":"test", 

    # "outputdict":{ # for ffmpeg
    #             # '-vcodec': 'mpeg4',  #  high fps low res
    #             "-vcodec": "libx264",   #   low fps high res
    #             '-crf': '0',
    #             '-preset': 'slow',  # TODO check this
    #             '-pix_fmt': 'yuvj444p',
    #             "-framerate": "10", #   output video framerate 
    #             # TODO this doesnt work FPS
    #         },
    


class Calibration():
    def __init__(self, calibration_data=None, plot=False):
        """
            Fits some calibration data with a line and uses the fit 
            to correct voltage data.

            :param calibration_data: pd.DataFrame with readout from calibration .csv. 
                    Should have voltage at a set of weights for each sensor in the forceplate.
            :param plot: bool, if true the results of the fit are displayed.
        """
        self.calibration_data=calibration_data
        self.fitted, weights, voltages = self.fit_calibration()

        if plot:
            self.plot_fitted(weights, voltages)

    def parse_calibration_data(self):
        """
            Parses data from a spreadsheet with calibration data 
            to get the voltage at each weight for each channel.
        """
        sensors = set(self.calibration_data.Sensor.values)
        weights = list(set(self.calibration_data.weight.values))

        voltages = {ch:self.calibration_data.loc[self.calibration_data.Sensor == ch].voltage.values \
                        for ch in sensors} 
        return sensors, weights, voltages

    def fit_calibration(self):
        sensors, weights, voltages = self.parse_calibration_data()
        fitted = {}
        for ch in sensors:
            fitted[ch] = np.poly1d(np.polyfit(voltages[ch], weights, 1))
        return fitted, weights, voltages

    def plot_fitted(self, weights, voltages):    
        f, ax = plt.subplots() 
        for ch, voltages in voltages.items():
            ax.scatter(voltages, weights, label=ch, s=50, alpha=.5)

            x = np.linspace(0, np.max(voltages), 100)
            y = self.fitted[ch](x)
            ax.plot(x, y, alpha=.5, lw=2, ls='--')
        ax.set(title="calibration curve", xlabel="voltage", ylabel="weight (g)")
        ax.legend()

    def correct_raw(self, voltages, ch):
        return self.fitted[ch](voltages)



if __name__ == "__main__":
    calibration_file = 'D:\\Egzona\\Forceplate\\forceplatesensors_calibration4.csv'
    calibration_data = load_csv_file(calibration_file)
    c = Calibration(calibration_data=calibration_data, plot=True)
    plt.show()