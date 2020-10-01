import sys
sys.path.append("./")

from pyinspect import install_traceback, search
install_traceback(keep_frames=1)

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime

from fcutils.file_io.utils import check_file_exists, check_file_exists, check_create_folder
from fcutils.file_io.io import load_csv_file, save_json

from analysis.utils.analysis_utils import parse_folder_files
from analysis.utils.utils import calibrate_sensors_data, correct_paw_used, compute_center_of_gravity



# ---------------------------------------------------------------------------- #
#                                     SETUP                                    #
# ---------------------------------------------------------------------------- #

# --------------------------------- Variables -------------------------------- #
fps = 600
n_frames = 500 # Number of frames to take after the "start" of the trial

calibrate_sensors = True
weight_percent = True # If calibrate_sensors is True and this is true traces are
                            # represent percentage of the animals weight
correct_for_paw = True # If true trials with both paws are used, after correcting for paw
            # Otherwise select trials with which paw to use
use_paw = 'R'


# ----------------------------------- Files ---------------------------------- #
frames_file = "D:\\Egzona\\Forceplate\\clipsframes_FP8.csv"
calibration_file = "D:\\Egzona\\Forceplate\\forceplatesensors_calibration4.csv"

# Folders to analyse
main_fld = "D:\\Egzona\\Forceplate\\2020"
sub_flds = {"21":os.path.join(main_fld, "21012020"),
            "23":os.path.join(main_fld, "23012020"),
            "24":os.path.join(main_fld, "24012020"),
            "28":os.path.join(main_fld, "28012020"), 
            "29":os.path.join(main_fld, "29012020"),
            "27":os.path.join(main_fld, "27022020"),
            "02":os.path.join(main_fld, "02032020NEW"),
            "03":os.path.join(main_fld, "03032020NEW"),
            "06":os.path.join(main_fld, "060720(training)"),
            "08":os.path.join(main_fld, "080720(training)"), 
            "10":os.path.join(main_fld, "100720(training)"),
            "20":os.path.join(main_fld, "200720(training)")} 
        

# Excel spreadsheets with start frame for each trial
framesfile = os.path.join(main_fld, "clipsframes_FP4.csv")

# Save path
savepath = os.path.join(main_fld, "data.hdf")
metadata_savepath = os.path.join(main_fld, "metadata.json")

# ----------------------------------- Misc ----------------------------------- #
sensors = ["fr", "fl", "hr", "hl"]

# ---------------------------------- Checks ---------------------------------- #
for subfold in sub_flds.values(): 
    check_create_folder(subfold, raise_error=True)
check_file_exists(frames_file, raise_error=True)

frames_data = pd.read_csv(frames_file)

if calibrate_sensors:
    check_file_exists(calibration_file, raise_error=True)
    calibration_data = load_csv_file(calibration_file)
    

# ---------------------------------------------------------------------------- #
#                                 PROCESS DATA                                 #
# ---------------------------------------------------------------------------- #

# Load data for each video
data = {"name":[], "fr":[], "fl":[], "hr":[], "hl":[], "CoG":[], "centered_CoG":[], "start":[], "end":[]}
for i, trial in tqdm(frames_data.iterrows()):
    # --------------------------- Fetch files for trial -------------------------- #
    if trial.Video[:2] not in list(sub_flds.keys()): 
        raise ValueError(f"Can't find a subfolder for trial with video name\n {trial}.\n Check your frames spreadsheet.")

    csv_file, video_files = parse_folder_files(sub_flds[trial.Video[:2]], trial.Video)

    # Load and trim sensors data
    start_frame, end_frame = trial.Start, trial.Start + n_frames
    sensors_data = load_csv_file(csv_file)
    sensors_data = {ch:sensors_data[ch].values for ch in sensors}
    sensors_data = {ch:v[start_frame:end_frame] for ch,v in sensors_data.items()}
 

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

    # compute center of gravity
    CoG, centered_CoG = compute_center_of_gravity(sensors_data)

    # Organise data
    data["name"].append(trial.Video)
    for ch, vals in sensors_data.items():
        data[ch].append(vals)

    data["CoG"].append(CoG)
    data["centered_CoG"].append(centered_CoG)
    data["start"].append(start_frame)
    data["end"].append(end_frame)

data = pd.DataFrame.from_dict(data)
if not len(data):
    print("No data was loaded, something went wrong!")
else:
    print("\nLoaded data")


print("Saving data to: {}".format(savepath))
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
print("Saving metadata to: {}".format(metadata_savepath))
save_json(metadata_savepath, metadata)



