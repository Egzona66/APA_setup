import sys
sys.path.append("./")

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import resample

from utils.video_utils import Editor as VideoUtils
from forceplate_config import Config
from utils.maths.filtering import line_smoother


from utils.file_io_utils import *
from utils.analysis_utils import *
from utils.video_utils import Editor as VideoUtils
from utils.matplotlib_config import *
from utils.plotting_utils import *
from utils.constants import *

class Calibration(Config):
    def __init__(self):
        Config.__init__(self)
        self.calibrate_sensors(plot=True)

    def calibrate_sensors(self, plot=False):
        calibration_data = load_csv_file("D:\\Egzona\\forceplatesensors_calibration.csv")

        readouts = dict(fr=[], fl=[], hr=[], hl=[])
        weights = []
        for i, row in calibration_data.iterrows():
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