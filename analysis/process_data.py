from loguru import logger
from pathlib import Path
import pandas as pd
import numpy as np

from fcutils.path import read_yaml, subdirs
from fcutils.progress import track
from fcutils.math.signals import convolve_with_gaussian, get_onset_offset
from fcutils.math.array import resample_list_of_arrayes_to_avg_len
from tpd import recorder

from analysis.calibration import Calibration
from analysis import utils
from analysis.fixtures import sensors


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
        "condition": [],
        "tot_weight": [],
        "on_sensors": [],
    }

    def __init__(self):
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
        processor = DataProcessing()
        params = read_yaml("./logs/params.yaml")

        logger.debug(f"Setting previously stored params: {params}")
        processor.load_set_params(params)

        # load data
        logger.info(f"Loading previously saved data from: {processor.data_savepath}")
        processor.data = pd.read_hdf(processor.data_savepath, key="hdf")

    def load_set_params(self, params: dict = None):
        # load parameters and set to class attributes
        params = params or read_yaml("./analysis/params.yaml")
        for name, param in params.items():
            if isinstance(param, str):
                setattr(self, name, Path(param))
            else:
                setattr(self, name, param)

        logger.info(
            f"Starting data pre-processing with trials file: {self.frames_file}."
        )
        logger.info(f"Caliration file: {self.calibration_file}.")

        if not self.frames_file.exists() or not self.calibration_file.exists():
            raise ValueError("Frames or calibration files not found!")
        recorder.copy("./analysis/params.yaml")

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
        target_n_samples = n_seconds * self.fps

        for k, v in sensors_data.items():
            adjusted = resample_list_of_arrayes_to_avg_len([v], target_n_samples)
            sensors_data[k] = adjusted
        return sensors_data

    def compute_sensors_engagement(self, sensors_data: dict) -> dict:
        """
            Computs when each sensor has weight on it and when
            the mouse is on all sensors
        """
        # get when mouse on each sensor
        paws_on_sensors = {
            f"{paw}_on_sensor": (sensors_data[paw] > self.on_sensor_weight_th).astype(
                np.int
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
        ).astype(np.int)
        sensors_data["on_sensors"] = (
            sensors_data["weight_on_sensors"] & sensors_data["all_paws_on_sensors"]
        ).astype(np.int)

        return sensors_data

    def process_trials(self):
        """
            Loads each trial's data, calibrates and filters the sensors data and then checks
            if the trial meets the criteria for analysis. If it does it's added to the dataset.
        """
        for i, trial in track(self.trials_metadata, total=len(self.trials_metadata)):
            logger.debug(f"Processing trial: {trial}")

            # fetch data
            csv_file, video_files = utils.parse_folder_files(
                self.main_fld / trial.subfolder, trial.Video
            )
            sensors_data = pd.read_csv(csv_file)
            sensors_data = {ch: sensors_data[ch] for ch in sensors}
            logger.debug(
                f"Loading senors data from CSV file: {csv_file.name} for tiral: {trial}"
            )

            # resample data to match target FPS
            if trial.fps != self.fps:
                sensors_data = self._resample_data(sensors_data, trial.fps)

            # smooth signals
            kernel_width = int(self.fps * self.smoothing_window)
            for k, v in sensors_data:
                sensors_data[k] = convolve_with_gaussian(v, kernel_width)

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

            # get trial start (and trial bounds)
            start_frame = int((trial.Start / trial.fps) * self.fps)
            if self.STANDING_STILL:
                on_sensors = start_frame
            else:
                on_sensors = get_onset_offset(sensors_data["on_sensors"][:start_frame])[
                    0
                ][-1]
            trial_start = start_frame - int(self.n_secs_before * self.fps)
            trial_end = start_frame + int(self.n_secs_after * self.fps)

            # check that trial matches criteria
            if (start_frame - on_sensors) / self.fps < self.min_baseline_duration:
                logger.warning("Basleline too short, excluding trial")
                continue
            if not sensors_data["on_sensors"][start_frame]:
                logger.warning("Mouse not on sensors at trial start, excluding trial.")
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

            self.data["CoG"].append(utils.compute_cog(sensors_data))
            self.data["condition "].append(trial.Condition)

        self.data = pd.DataFrame(self.data)

    def wrapup(self):
        """
            Save data
        """
        self.data.to_hdf(self.data_savepath, key="hdf")
        logger.info(f"Saving data for {len(self.data)} trials at: {self.data_savepath}")


if __name__ == "__main__":
    DataProcessing().process_trials()
