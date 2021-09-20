import pandas as pd
import numpy as np
from typing import List, Tuple
from loguru import logger

from tpd import recorder
import matplotlib.pyplot as plt

from analysis.fixtures import sensors, colors
from analysis.utils import convolve_with_gaussian


class Calibration:
    fit_degree = 1

    def __init__(self, calibration_data: pd.DataFrame):
        """
            Fits a curve of degree `fit_degree` to the calibration
            data which consists fo voltage-grams pairs for each sensor.
        """
        self.calibration_data: pd.DataFrame = calibration_data
        self.weights, self.voltages = self.parse_calibration_data()
        self.fitted_curves = self.fit()

        self.plot_fit()
        logger.info("Calibration fitted to calibrationd data")

    def parse_calibration_data(self) -> Tuple[List, dict]:
        """
            Parses data from a spreadsheet with calibration data 
            to get the voltage at each weight for each channel.
        """
        weights = list(set(self.calibration_data.weight.values))
        voltages = {
            ch: self.calibration_data.loc[
                self.calibration_data.Sensor == ch
            ].voltage.values
            for ch in sensors
        }
        return weights, voltages

    def fit(self) -> dict:
        """
            Fits a polynomial of degree fit_degree to the data
        """
        fitted = {}
        for ch in sensors:
            fitted[ch] = np.poly1d(
                np.polyfit(self.voltages[ch], self.weights, self.fit_degree)
            )
        return fitted

    def plot_fit(self):
        """
            Plots the weights and fits fo each sensor
        """
        f, ax = plt.subplots(figsize=(9, 9))
        f._save_name = "calibration_data_fit"

        for ch, voltages in self.voltages.items():
            ax.scatter(
                voltages,
                self.weights,
                label=ch,
                s=150,
                color="w",
                ec=colors[ch],
                lw=1,
                zorder=100,
            )

            x = np.linspace(0, np.max(voltages), 100)
            y = self.fitted[ch](x)
            ax.plot(x, y, alpha=0.5, lw=2, color=colors[ch])
        ax.set(title="calibration curve", xlabel="voltage", ylabel="weight (g)")
        ax.legend()

        recorder.add_figure(f)
        plt.close(f)

    def correct_raw(self, voltages: np.ndarray, ch: str) -> np.ndarray:
        return self.fitted[ch](voltages)

    def calibrate(
        self, sensors_data: dict, weight_percentage: bool, mouse_weight: float
    ) -> dict:
        """
            Uses the calibration curves to go from voltages -> grams.
            If weight_percentage is true then the values in grams are converted to
            % of mouse body weight
        """
        logger.debug(
            f"Calibrating data - (as %: {weight_percentage} | mouse weight: {mouse_weight}g)"
        )
        # compute and subtract baselines to 0 data correctly
        for ch, data in sensors_data.items():
            baseline = np.percentile(convolve_with_gaussian(data, 600)[1000:-1000], 1)
            sensors_data[ch] = data - baseline

        # volts -> grams
        calibrated = {
            ch: self.correct_raw(volts, ch)
            for ch, volts in sensors_data.items()
            if ch in sensors
        }

        # grams -> %
        if weight_percentage:
            calibrated = {ch: grams / mouse_weight for ch, grams in calibrated.items()}

        return calibrated
