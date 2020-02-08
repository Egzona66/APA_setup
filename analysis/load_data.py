# %%
import sys
sys.path.append("./")

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from fcutils.file_io.utils import check_file_exists, check_file_exists, check_create_folder
from fcutils.file_io.io import load_csv_file

from utils.analysis_utils import parse_folder_files
from utils.utils import calibrate_sensors_data, compute_center_of_gravity
%matplotlib inline

# %%
# ---------------------------------------------------------------------------- #
#                                     SETUP                                    #
# ---------------------------------------------------------------------------- #

# --------------------------------- Variables -------------------------------- #
fps = 600
n_frames = 400 # Number of frames to take after the "start" of the trial

calibrate_sensors = False
correct_for_paw = False # If true trials with both paws are used, after correcting for paw
            # Otherwise select trials with which paw to use
use_paw = 'R'


# ----------------------------------- Files ---------------------------------- #
frames_file = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Egzona/clipsframes_FP3.csv"
calibration_file = '/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Egzona/forceplatesensors_calibration2.csv'

# Folders to analyse
main_fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Egzona/2020"
sub_flds = {"21":os.path.join(main_fld, "21012020"),
            "23":os.path.join(main_fld, "23012020"), 
            "24":os.path.join(main_fld, "24012020"), 
            "28":os.path.join(main_fld, "28012020"), 
            "29":os.path.join(main_fld, "29012020")} 

# Excel spreadsheets with start frame for each trial
framesfile = os.path.join(main_fld, "clipsframes_FP3.csv")

# Save path
savepath = os.path.join(main_fld, "data.hdf")

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
    


# %%
# ---------------------------------------------------------------------------- #
#                                   LOAD DATA                                  #
# ---------------------------------------------------------------------------- #

# Load data for each video
data = {"name":[], "fr":[], "fl":[], "hr":[], "hl":[], "CoG":[], "centered_CoG":[], "start":[], "end":[]}
for i, trial in tqdm(frames_data.iterrows()):
    # --------------------------- Fetch files for trial -------------------------- #
    if trial.Video[:2] not in list(sub_flds.keys()): 
        raise ValueError("Can't find a subfolder for trial with video name {}.\n Check your frames spreadsheet.")

    csv_file, video_files = parse_folder_files(sub_flds[trial.Video[:2]], trial.Video)
    if csv_file is None: 
        raise ValueError("cvs file is None")

    # Load and trim sensors data
    start_frame, end_frame = trial.Start, trial.Start + n_frames
    sensors_data = load_csv_file(csv_file)
    sensors_data = {ch:sensors_data[ch].values for ch in sensors}
    sensors_data = {ch:v[start_frame:end_frame] for ch,v in sensors_data.items()}

    # Get baselined and calibrated sensors data
    if calibrate_sensors:
        sensors_data = calibrate_sensors_data(sensors_data, sensors, calibration_data=calibration_data)

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



# %%
