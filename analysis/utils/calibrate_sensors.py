import sys
sys.path.append("./")

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import resample

from utils.analysis_utils import *

from fcutils.maths.filtering import line_smoother
from fcutils.video.video_editing import Editor as VideoUtils
from fcutils.file_io.utils import *
from fcutils.file_io.io import load_csv_file
from fcutils.plotting.utils import *
from fcutils.plotting.colors import red, blue, green, pink, magenta, white


analysis_config = {
    "plot_colors": { "fr":magenta, 
                    "fl":blue}, 
                    #"hr":red, 
                    #"hl":green},

    # * for composite video
    # ? run video_analysis.py
    "start_clip_time_s": None, # ? Create clips start at this point, in SECONDS
    "start_clip_time_frame": 9799, # ? Create clips start at this point, in FRAMES
    "clip_n_frames": 180 , # duration of the clip in frames
    "clip_name":"test", 

    "outputdict":{ # for ffmpeg
                # '-vcodec': 'mpeg4',  #  high fps low res
                "-vcodec": "libx264",   #   low fps high res
                '-crf': '0',
                '-preset': 'slow',  # TODO check this
                '-pix_fmt': 'yuvj444p',
                "-framerate": "10", #   output video framerate 
                # TODO this doesnt work FPS
            },
    }


class Calibration():
    def __init__(self, calibration_data=None, plot=False):
        self.calibration_data=calibration_data
        self.calibrate_sensors(plot=plot)


    def calibrate_sensors(self, plot=False):
        if self.calibration_data is None:
            calibration_data = load_csv_file("D:\\Egzona\\forceplatesensors_calibration2.csv")
        else:
            calibration_data = self.calibration_data

        readouts = dict(fr=[], fl=[], hr=[], hl=[])
        weights = []
        for i, row in calibration_data.iterrows():
            # Get the average of the voltage readings
            measurements = [row[k] for k in row.keys() if "voltage" in k and row[k]]
            voltage = np.nanmean(measurements)
            readouts[row.Sensor].append(row.voltage/5)

            if row.Sensor == "fr":
                weights.append(row.weight)

        if plot:
            f, ax = plt.subplots() 
            fits = {}
            for ch, voltages in readouts.items():
                fit = np.polyfit(voltages, weights,  6)
                fitplot = np.poly1d(fit)
                fits[ch] = fitplot

                x = np.linspace(0, .4, 100)
                ax.scatter(voltages, weights, color=self.analysis_config["plot_colors"][ch], label=ch, s=100)
                ax.plot(x, fitplot(x), color=self.analysis_config["plot_colors"][ch], label=ch)
            ax.set(title="calibration curve", xlabel="voltage", ylabel="weight (g)")
            ax.legend()
        else:
            fits = {}
            for ch, voltages in readouts.items():
                fit = np.polyfit(voltages, weights,  6)
                fitplot = np.poly1d(fit)
                fits[ch] = fitplot

        self.calibration_funcs = fits
        return fits

    def correct_raw(self, voltages, ch, calibration_funcs=None):
        if calibration_funcs is not None: raise NotImplementedError
        else:
            return self.calibration_funcs[ch](voltages)

    def test(self):
        self.calibrate_sensors(plot=True)

        x = np.arange(0, 3)
        xcorr = self.correct_raw(x, "fr")

        plt.plot(x, xcorr, color=white, lw=4, alpha=.4, label="correct")
        plt.legend()





if __name__ == "__main__":
    c = Calibration()
    plt.show()