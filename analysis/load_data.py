import sys
sys.path.append("./")

from pyinspect import install_traceback, search
install_traceback(keep_frames=0, hide_locals=True)

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from loguru import logger
from pathlib import Path
from rich.progress import track

from fcutils.file_io.utils import check_file_exists, check_file_exists, check_create_folder
from fcutils.file_io.io import load_csv_file, save_json
from fcutils.maths.utils import rolling_mean

from analysis.utils.analysis_utils import parse_folder_files
from analysis.utils.utils import calibrate_sensors_data, correct_paw_used, compute_center_of_gravity, get_onset_offset
from analysis import paths

# ---------------------------------------------------------------------------- #
#                                     SETUP                                    #
# ---------------------------------------------------------------------------- #
DEBUG = False  # set as true to have extra plots to check everything's OK

# --------------------------------- Variables -------------------------------- #
CONDITIONS = ('WT', )  # keep only data from these conditions

fps = 600
n_frames = 1500 # Number of frames to take after the "start" of the trial

calibrate_sensors = True
weight_percent = True # If calibrate_sensors is True and this is true traces are
                            # represent percentage of the animals weight
correct_for_paw = True # If true trials with both paws are used, after correcting for paw
            # Otherwise select trials with which paw to use
use_paw = 'R'

if not weight_percent or not correct_for_paw:
    raise ValueError('Fede asked you not to change weight_percent and correct_for_paw options')

sensors = ["fl", "fr", "hl", "hr"]

frames_file = "D:\\Egzona\\Forceplate\\ALL_APA_trials_2021_new_analysis.csv"
calibration_file = "D:\\Egzona\\Forceplate\\forceplatesensors_calibration4.csv"

# Folders to analyse
main_fld = Path("D:\\Egzona\\Forceplate\\2021")
sub_flds = paths.subdirs(main_fld)
logger.debug(f'Found {len(sub_flds)} subfolders in main folder: {main_fld}')

# Save path
savepath = os.path.join(main_fld, "data.hdf")
metadata_savepath = os.path.join(main_fld, "metadata.json")



# ---------------------------------------------------------------------------- #
#                                     RUN                                      #
# ---------------------------------------------------------------------------- #

def run():


    for fpath in (frames_file, calibration_file):
        if not Path(fpath).exists():
            raise FileNotFoundError(f'Could not find file: {fpath}')
        
    # load frames data and match experiments to subfolders
    frames_data = pd.read_csv(frames_file)

    # ----------------------------------- Data  ---------------------------------- #
    def clean(string):
        return string.split('_M')[0].split('_F')[0]
    frames_data['subfolder'] = frames_data.Video.apply(clean)

    # check that all experiments sufolers are found
    subfolds_names = [fld.name for fld in sub_flds]
    if not np.all(frames_data.subfolder.isin(subfolds_names)):
        errors = frames_data.loc[~frames_data.subfolder.isin(subfolds_names)]
        raise ValueError(f'At least one subfolder from the frames spreadsheet was not found in the subfolders of {main_fld}:\n{errors}')

    # filter data by condition
    frames_data = frames_data.loc[frames_data.Condition.isin(CONDITIONS)]
    logger.debug(f'Got {len(frames_data)} trials data:\n{frames_data.head()}')

    # load calibration data
    calibration_data = load_csv_file(calibration_file)
        
    logger.debug('All checks passed and all files found')
    # ---------------------------------------------------------------------------- #
    #                                 PROCESS DATA                                 #
    # ---------------------------------------------------------------------------- #

    # Load data for each video
    data = {"name":[], "fr":[], "fl":[], "hr":[], "hl":[], "CoG":[], "centered_CoG":[], "start":[], "end":[], 'condition':[], 'fps':[]}
    for i, trial in track(frames_data.iterrows(), total=len(frames_data)):
        keep = True  # to be changed if trial is BAD
        # --------------------------- Fetch files for trial -------------------------- #=
        csv_file, video_files = parse_folder_files(main_fld / trial.subfolder, trial.Video)
        logger.info(f'Found csv file: {csv_file}')

        # Load and trim sensors data
        sensors_data = load_csv_file(csv_file)

        sensors_data = {ch:rolling_mean(sensors_data[ch], 60) for ch in sensors}

        # debug plots: RAW data
        # if DEBUG:
        #     f, ax = plt.subplots(figsize=(16, 9))
        #     ax.set(title=trial.Video)


        #     colors = 'rgbm'
        #     for sens, col in zip(sensors, colors):
        #         ax.plot(sensors_data[sens], label=sens, color=col)
        #     ax.axvline(trial.Start, lw=3, color='k', label='start')

        #     baselines = dict(
        #         fr = trial.baselineFR,
        #         fl = trial.baselineFL,
        #         hr = trial.baselineHR,
        #         hl = trial.baselineHL,
        #     )
        #     for col, (ch, bl) in zip(colors, baselines.items()):
        #         ax.axhline(bl, label=f'Sensors {ch}', color=col)


        #     ax.legend()
        #     plt.show()

        # Get baselined and calibrated sensors data
        if calibrate_sensors:
            sensors_data = calibrate_sensors_data(sensors_data, sensors, 
                                            calibration_data=calibration_data, 
                                            weight_percent=weight_percent,
                                            mouse_weight=trial.Weight,
                                            direction=trial.Direction, paw=trial.Paw, 
                                            base_voltageFR=trial.baselineFR, base_voltageFL=trial.baselineFL,
                                            base_voltageHR=trial.baselineHR, base_voltageHL=trial.baselineHL)


        # Check paw used or skip wrong paw trials
        if correct_for_paw:
            sensors_data = correct_paw_used(sensors_data, trial.Paw)
        elif trial.Paw.upper() != use_paw:
            continue

        # check when all paws are on sensors
        paws_on_sensors = {f'{paw}_on_sensor': (sensors_data[paw] > 6).astype(np.int) for paw in sensors}
        all_on_sensors = np.sum(np.vstack(list(paws_on_sensors.values())), 0)
        all_on_sensors[all_on_sensors < 4] = 0
        all_on_sensors[all_on_sensors == 4] = 1
        paws_on_sensors['all_paws_on_sensors'] = all_on_sensors
        sensors_data.update(paws_on_sensors)


        # get comulative weight on sensors    
        sensors_data['tot_weight'] = np.sum(np.vstack([sensors_data[p] for p in sensors]), 0)
        sensors_data['weight_on_sensors'] = (sensors_data['tot_weight'] > 80).astype(np.int) 
        sensors_data['on_sensors'] = (sensors_data['weight_on_sensors'] & sensors_data['all_paws_on_sensors']).astype(np.int)
        
        # get trial start (last time on_sensors == 1 before trial.Start)
        start = get_onset_offset(sensors_data['on_sensors'][:trial.Start], .5)[0][-1]
        end_frame = trial.Start + n_frames

        # remove trials where conditions are wrong
        baseline_duration = np.abs((trial.Start - start)/trial.fps)
        if baseline_duration > 5 or baseline_duration < .2:
            logger.warning(f'Excluding trial: {trial.Video} because the baseline was either too long or too short: {round(baseline_duration, 3)}s')
            keep=False
        if not sensors_data['on_sensors'][trial.Start - int(.2 * trial.fps)]:
            logger.warning(f'Excluding trial: {trial.Video} because at trial.Start the conditions were not met, sorry')
            keep=False

        # debug plots: CALIBRATE data
        if DEBUG:
            f, ax = plt.subplots(figsize=(16, 9))
            ax.set(title=trial.Video + f'   Is it good: {keep}')

            colors = 'rgbm'
            for sens, col in zip(sensors, colors):
                ax.plot(sensors_data[sens], label=sens, color=col)

            ax.plot(sensors_data['tot_weight'], label='tot', color='k', lw=3, zorder=-1)
            ax.plot((sensors_data['on_sensors'] * 10) + 100,  lw=8, color=[.4, .4, .4])

            ax.axvline(trial.Start, lw=3, color='k', label='manual start')
            ax.axvline(end_frame, lw=2, color='k', label='manual end')
            ax.axvline(start, lw=3, color='r', label='trial start')

            ax.set(xlim=[start-2000, start+2000])
            ax.legend()
            plt.show()
        if not keep: continue

        # cut trial
        sensors_data = {ch:v[start:end_frame] for ch,v in sensors_data.items()}

        # compute center of gravity
        CoG, centered_CoG = compute_center_of_gravity(sensors_data)
        logger.debug('Finished "correcting" sensors data')

        # Organise data
        data["name"].append(trial.Video)
        for ch, vals in sensors_data.items():
            if ch not in data.keys():
                data[ch] = []
            data[ch].append(vals)

        data["CoG"].append(CoG)
        data["centered_CoG"].append(centered_CoG)
        data["start"].append(start)
        data["end"].append(end_frame)
        data['condition'].append(trial.Condition)
        data['fps'].append(trial.fps)

    data = pd.DataFrame.from_dict(data)
    if not len(data):
        raise ValueError("No data was loaded, something went wrong!")
    else:
        logger.info(f"\n\n\n=========\nLoaded data for {len(data)} trials, yay!\n=========\nCount:\n")
        logger.info(data.groupby('condition')['name'].count())


    logger.info("Saving data to: {}".format(savepath))
    data.to_hdf(savepath, key='hdf')

    # ------------------------------- Save metadata ------------------------------ #
    metadata = dict(
        date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        fps=fps,
        n_frames=n_frames,
        calibrate_sensors=calibrate_sensors,
        weight_percent=weight_percent,
        correct_for_paw=correct_for_paw,
        use_paw=use_paw,
        frames_file=frames_file,
        sensors=sensors,
    )
    logger.info("Saving metadata to: {}".format(metadata_savepath))
    save_json(metadata_savepath, metadata)


if __name__ == "__main__":
    run()
