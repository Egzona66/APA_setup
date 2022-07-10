from loguru import logger
from pathlib import Path
import pandas as pd
import numpy as np

import sys

sys.path.append("./")
sys.path.append(r"C:\Users\ucqfajm\Documents\GitHub\APA_setup\analysis")
from fcutils.path import from_yaml, subdirs
from fcutils.progress import track
from fcutils.maths.signals import convolve_with_gaussian, get_onset_offset
from fcutils.maths.array import resample_list_of_arrayes_to_avg_len
from tpd import recorder
from fcutils.path import files

from analysis.calibration import Calibration
from analysis import utils
from analysis.fixtures import sensors

# from analysis.debug import plot_sensors_data


def clean(string: str) -> str:
    return string.split("_M")[0].split("_F")[0]


class DataProcessing:
    data = {
        "name": [],
        "fr": [],
        "fl": [],
        "hr": [],
        "hl": [],
        "CoG": [],
        "CoG_centered": [],
        "condition": [],
        "strain": [],
        "tot_weight": [],
        "on_sensors": [],
        "movement_onset_frame": [],
        "video": [],
        "original_fps":[],
    }

    def __init__(self, reloading=False):
        """
            Loads trials metadata, processes the trials' sensors data
            and filters trials based on them meeting criteria for analysis.
        """
        self.load_set_params()
        # get trials subplots
        self.trials_dirs = subdirs(self.main_fld)
        logger.info(f"Found {len(self.trials_dirs)} subfolders")

        # Save path
        self.data_savepath = self.main_fld / "data.h5"

        if not reloading:
            # load trials data and get subfolders name
            self.trials_metadata = pd.read_csv(self.frames_file)
            self.trials_metadata["subfolder"] = self.trials_metadata.Video.apply(clean)
            logger.info(f"Found metadata for {len(self.trials_metadata)} trials.")

            # filter trials based on conditions to analyze
            self.trials_metadata = self.trials_metadata.loc[
                self.trials_metadata.Condition.isin(self.CONDITIONS)
            ]
            logger.info(
                f"Keeping {len(self.trials_metadata)} for conditions: {self.CONDITIONS}"
            )

            # check data
            self.preliminary_checks()

            # prepare calibration
            self.calibration_util = Calibration(pd.read_csv(self.calibration_file))

    @classmethod
    def reload(cls):
        """
            Returns an instance of DataProcessing with loaded
            pre-processed data
        """
        # initialize and retrieve params used for previous processing
        processor = DataProcessing(reloading=True)
        params = from_yaml("./logs/params.yaml")
        

        logger.debug(f"Setting previously stored params: {params}")
        processor.load_set_params(params)
        processor.params = params

        # load data
        logger.info(f"Loading previously saved data from: {processor.data_savepath}")
        data = pd.read_hdf(processor.data_savepath, key="hdf")
        # filter by strain/condition
        data = data.loc[data.condition.isin(params["CONDITIONS"])].reset_index(drop=True)
        data = data.loc[data.strain.str.upper().isin(params["STRAINS"])].reset_index(drop=True)
        
        processor.data = data
        logger.info(f"Loaded {len(processor.data)} trials -----\n\n")

        return processor

    def load_set_params(self, params: dict = None):
        # load parameters and set to class attributes
        try:
            params = params or from_yaml("./analysis/params.yaml")
        except:
            params = from_yaml("./params.yaml")
        

        for name, param in params.items():
            if isinstance(param, str):
                setattr(self, name, Path(param))
            else:
                setattr(self, name, param)

        logger.info(
            f"Starting data pre-processing with trials file: {self.frames_file}."
        )
        logger.info(f"Caliration file: {self.calibration_file}.")

        if not self.frames_file.exists():
            raise ValueError(f"Frames files not found!\n{self.frames_file}")

        if not self.calibration_file.exists():
            raise ValueError(f"Calibration files not found!\n{self.calibration_file}")

    def preliminary_checks(self):
        """
            Run basic preliminary checks
        """
        # check that all experiments sufolers are found
        subfolds_names = [fld.name for fld in self.trials_dirs]
        if not np.all(self.trials_metadata.subfolder.isin(subfolds_names)):
            errors = self.trials_metadata.loc[
                ~self.trials_metadata.subfolder.isin(subfolds_names)
            ]
            raise ValueError(
                f"At least one subfolder from the frames spreadsheet was not found in the subfolders of {self.main_fld}:\n{errors}"
            )

    def _resample_data(self, sensors_data: dict, originalfps: int) -> dict:
        """
            Resamples sensors data to match the target fps
        """
        n_seconds = len(sensors_data["fr"]) / originalfps
        target_n_samples = int(n_seconds * self.fps)

        for k, v in sensors_data.items():
            adjusted = resample_list_of_arrayes_to_avg_len([v], target_n_samples)
            sensors_data[k] = adjusted.T.ravel()
        return sensors_data

    def compute_sensors_engagement(self, sensors_data: dict) -> dict:
        """
            Computs when each sensor has weight on it and when
            the mouse is on all sensors
        """
        # get when mouse on each sensor
        paws_on_sensors = {
            f"{paw}_on_sensor": (sensors_data[paw] > self.on_sensor_weight_th).astype(
                np.int64
            )
            for paw in sensors
        }
        all_on_sensors = np.sum(np.vstack(list(paws_on_sensors.values())), 0)
        all_on_sensors[all_on_sensors < 4] = 0
        all_on_sensors[all_on_sensors == 4] = 1
        paws_on_sensors["all_paws_on_sensors"] = all_on_sensors
        sensors_data.update(paws_on_sensors)

        # get comulative weight on sensors
        sensors_data["tot_weight"] = np.sum(
            np.vstack([sensors_data[p] for p in sensors]), 0
        )
        sensors_data["weight_on_sensors"] = (
            sensors_data["tot_weight"] > self.on_all_sensors_weight_th
        ).astype(np.int64)
        sensors_data["on_sensors"] = (
            sensors_data["weight_on_sensors"] & sensors_data["all_paws_on_sensors"]
        ).astype(np.int64)

        return sensors_data

    def process_trials(self):
        """
            Loads each trial's data, calibrates and filters the sensors data and then checks
            if the trial meets the criteria for analysis. If it does it's added to the dataset.
        """

        excluded = []
        for i, trial in track(
            self.trials_metadata.iterrows(), total=len(self.trials_metadata)
        ):
            # fetch data
            csv_file = files(
                self.main_fld / trial.subfolder, f"{trial.Video}_analoginputs.csv"
            )
            if csv_file is None:
                raise ValueError("Could not find CSV file for experiment!")
            elif isinstance(csv_file, list):
                raise ValueError("Found too many CSV files!!")

            sensors_data = pd.read_csv(csv_file)
            sensors_data = {ch: sensors_data[ch] for ch in sensors}
            logger.debug(
                f"Loading senors data from CSV file: {csv_file.name} for trial: {trial['Video']}"
            )

            # resample data to match target FPS
            if trial.fps != self.fps:
                sensors_data = self._resample_data(sensors_data, trial.fps)

            # smooth signals
            kernel_width = int(self.fps * self.smoothing_window)
            for k, v in sensors_data.items():
                sensors_data[k] = convolve_with_gaussian(v, kernel_width)
                sensors_data[k][:40] = sensors_data[k][-40:] = np.mean(sensors_data[k])

            # calibrate sensors
            if self.calibrate:
                sensors_data = self.calibration_util.calibrate(
                    sensors_data, self.weight_percentage, trial.Weight,
                )

            # correct moving paw
            if self.correct_for_paw:
                sensors_data = utils.correct_paw_used(sensors_data, trial.Paw)

            # get when mouse on sensors
            sensors_data = self.compute_sensors_engagement(sensors_data)

            # get trial start (first frame in which FL paw lifts off in a window around manually specified start)
            manual_start_frame = int((trial.Start / trial.fps) * self.fps)

            if not self.STANDING_STILL:
                half_window = int(self.trial_start_detection_window * self.fps)
                try:
                    start_frame = (
                        np.where(
                            sensors_data["fr_on_sensor"][
                                manual_start_frame
                                - half_window : manual_start_frame
                                + half_window
                            ]
                            == 0
                        )[0][0]
                        - 2
                    )
                except IndexError:
                    start_frame = manual_start_frame
                else:
                    start_frame += manual_start_frame - half_window
            else:
                start_frame = manual_start_frame

            trial_start = start_frame - int(self.n_secs_before * self.fps)
            trial_end = start_frame + int(self.n_secs_after * self.fps)

            # check that trial matches criteria
            if not self.STANDING_STILL:
                on_sensors = get_onset_offset(
                    sensors_data["on_sensors"][:start_frame], 0.5
                )[0][-1] + 1

                if (start_frame - on_sensors) / self.fps < self.min_baseline_duration:
                    logger.warning("Basleline too short, excluding trial")
                    excluded.append(trial["Video"])
                    continue

            if not sensors_data["on_sensors"][start_frame]:
                logger.warning("Mouse not on sensors at trial start, excluding trial.")
                excluded.append(trial["Video"])
                continue

            # cut data
            sensors_data = {
                ch: v[trial_start:trial_end] for ch, v in sensors_data.items()
            }

            # append data
            self.data["name"].append(trial.Video)
            for ch, v in sensors_data.items():
                if ch in self.data.keys():
                    self.data[ch].append(v)

            # add CoG (including centered at the CoG of t=0)
            self.data["CoG"].append(utils.compute_cog(sensors_data))
            self.data["CoG_centered"].append(self.data["CoG"][-1] - self.data["CoG"][-1][start_frame - trial_start, :])
            self.data["condition"].append(trial.Condition)
            self.data["strain"].append(trial.Strain)

            self.data["movement_onset_frame"].append(start_frame)
            self.data["video"].append(str(self.main_fld / trial.subfolder / f"{trial.Video}_cam0.avi"))
            self.data["original_fps"].append(trial.fps)

        self.data = pd.DataFrame(self.data)
        logger.info(f"\nExcluded {len(excluded)} trials: {excluded}")
        print("\n")
        print(self.data.groupby("condition").count()['name'])
        print("\n")
        print(self.data.groupby("strain").count()['name'])
        print("\n")

        self.wrapup()

    def wrapup(self):
        """
            Save data
        """
        self.data.to_hdf(self.data_savepath, key="hdf")
        logger.info(f"Saving data for {len(self.data)} trials at: {self.data_savepath}")

        recorder.copy("./analysis/params.yaml")


if __name__ == "__main__":
    DataProcessing().process_trials()
