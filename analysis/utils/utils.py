import sys
sys.path.append("./")

import numpy as np
from loguru import logger
import pandas as pd
from scipy.signal import resample
from scipy import stats
import matplotlib.pyplot as plt

from fcutils.maths.utils import derivative

from analysis.utils.calibrate_sensors import Calibration


def convolve_with_gaussian(data:np.ndarray, sigma:int):
    x = np.linspace(0.0001, 0.9999, num=sigma)
    norm = stats.norm(0, sigma)
    kernel = norm.pdf(x)
    kernel /= np.sum(kernel)

    return np.convolve(data, kernel, mode='same')

def resample_list_of_arrayes_to_avg_len(lst, N=None, interpolate=False):
    """
        Given a list of arrays of varying length, this function
        resamples them so that they all have the 
        average length.
        Then it returns the vstack of the array
    """
    if N is None:
        N = np.mean([len(x) for x in lst]).astype(np.int32)

    if interpolate:
        lst = [pd.Series(x).interpolate() for x in lst]

    return np.vstack([resample(X, N) for X in lst])


def get_onset_offset(signal, th, clean=True):
    """
        Get onset/offset times when a signal goes below>above and
        above>below a given threshold
        Arguments:
            signal: 1d numpy array
            th: float, threshold
            clean: bool. If true ends before the first start and 
                starts after the last end are removed
    """
    above = np.zeros_like(signal)
    above[signal >= th] = 1

    der = derivative(above)
    starts = np.where(der > 0)[0]
    ends = np.where(der < 0)[0]

    if above[0] > 0:
        starts = np.concatenate([[0], starts])
    if above[-1] > 0:
        ends = np.concatenate([ends, [len(signal)]])

    if clean:
        ends = np.array([e for e in ends if e > starts[0]])

        if np.any(ends):
            starts = np.array([s for s in starts if s < ends[-1]])

    if not np.any(starts):
        starts = np.array([0])
    if not np.any(ends):
        ends = np.array([len(signal)])

    return starts, ends


def baseline_sensor_data(values):
    return values - np.nanmedian(values)

# -------------------------------- Corrections ------------------------------- #

def calibrate_sensors_data(sensors_data, sensors, calibration_data=None,
                           weight_percent=False, mouse_weight=None,
                           direction=None, paw=None, base_voltageFR=None, 
                           base_voltageFL=None, base_voltageHR=None, base_voltageHL=None): 
    """
        Calibrates the sensors to convert voltages to grams

        :param sensors_data: dictionary with voltages at each frame for each channel
        :param sensors: list of strings with the name of allowed sensors
        :param calibration_data: data from csv file with calibration data
        :param weight_percent: if true the weights are expressed in percentage of the mouse weight
        :param mouse_weight: float, weight of the mouse whose data are being processed
    """

    # compute and subtract baselines
    baselines = {}
    for ch, data in sensors_data.items():
        baseline = np.percentile(convolve_with_gaussian(data, 600)[1000:-1000], 1)
        sensors_data[ch] = data - baseline

    # calibrate
    calibration = Calibration(calibration_data=calibration_data, plot=False)
    calibrated =  {ch:calibration.correct_raw(np.float32(volts), ch) 
                                for ch, volts in sensors_data.items() if ch in sensors}

    # Get the total weight on sensors and check if it's > mous weight
    tot_weight = convolve_with_gaussian(
        np.sum(np.vstack(calibrated.values()), 0)[1000:-1000],  # discard artifacts at start/end
        600
    )
    if np.any(tot_weight < -3):
        f, ax = plt.subplots()
        ax.plot(tot_weight, label='sensors weight', color='k')
        ax.axhline(mouse_weight, lw=2, color='salmon', label='mouse weight')
        ax.axhline(0, lw=2, color='green', label='0')
        ax.legend()
        calibration = Calibration(calibration_data=calibration_data, plot=True)

        f, ax = plt.subplots()
        for ch, data in sensors_data.items():
            ax.plot(data, label=ch)
        ax.legend()

        plt.show()
        raise ValueError('The ammount of weight on sensors is either negative or too large')

    if weight_percent:
        calibrated = {ch:(values/mouse_weight)*100 for ch, values in calibrated.items()}

    # for ch, corrected in calibrated.items():
    #     if np.any(corrected < 0):  # check that no negative values
    #             logger.warning(f'Channel: {ch}: after correction we got some negative values: {np.min(corrected):.3f}')

    return calibrated
     

def correct_paw_used(sensors_data, paw_used):
    if 'l' in paw_used.lower():
        return {
            'fr': sensors_data['fl'],
            'fl': sensors_data['fr'],
            'hr': sensors_data['hl'],
            'hl': sensors_data['hr']
        }
    return sensors_data


# ------------------------------- Computations ------------------------------- #
def compute_center_of_gravity(sensors_data):
    y = (sensors_data["fr"]+sensors_data["fl"]) - \
            (sensors_data["hr"]+sensors_data["hl"])
    x = (sensors_data["fr"]+sensors_data["hr"]) - \
            (sensors_data["fl"]+sensors_data["hl"])

    centered_x, centered_y = x-x[0], y-y[0]
    return np.vstack([x,y]).T, np.vstack([centered_x,centered_y]).T

