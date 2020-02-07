# %%
import sys
sys.path.append("./")

import pandas as pd
import numpy as np

from fcutils.file_io.utils import check_file_exists, check_file_exists
from fcutils.file_io.io import load_csv_file

from utils.analysis_utils import parse_folder_files
from analysis.utils.utils import calibrate_sensors_data, compute_center_of_gravity
%matplotlib inline

# %%
# ---------------------------------------------------------------------------- #
#                                     SETUP                                    #
# ---------------------------------------------------------------------------- #

# --------------------------------- Variables -------------------------------- #
fps = 600
n_frames = 200 # Number of frames to take after the "start" of the trial

calibrate_sensors = True
correct_for_paw = False # If true trials with both paws are used, after correcting for paw
            # Otherwise select trials with which paw to use
use_paw = 'L'


# ----------------------------------- Files ---------------------------------- #

# Folders to analyse
main_fld = "D:\\Egzona\\2020"
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

# Prepare a few things
frames_data = pd.read_csv('D:\Egzona\clipsframes_FP3.csv')

# ---------------------------------- Checks ---------------------------------- #
for subfold in sub_flds.values(): 
    check_file_exists(subfold, raise_error=True)
check_file_exists(framesfile, raise_error=True)



# %%
# ---------------------------------------------------------------------------- #
#                                   LOAD DATA                                  #
# ---------------------------------------------------------------------------- #

# Load data for each video
data = {"name":[], "fr":[], "fl":[], "hr":[], "hl":[], "cg":[], "start":[], "end":[]}
for i, trial in frames_data.iterrows():
    # --------------------------- Fetch files for trial -------------------------- #
    if trial.Video[:2] not in list(sub_flds.keys()): 
        raise ValueError("Can't find a subfolder for trial with video name {}.\n Check your frames spreadsheet.")

    csv_file, video_files = parse_folder_files(fld, trial.Video)
    if csv_file is None: 
        raise ValueError("cvs file is None")
    else:
        print("Loading file: {}  -- for video: {}".format(csv_file, trial.Video))

    # Load and trim sensors data
    start_frame, end_frame = trial.Start, trial.Start + n_frames
    sensors_data = load_csv_file(csv_file)
    sensors_data = {ch:v[start_frame:end_frame]}

    # Get baselined and calibrated sensors data
    if calibrate_sensors_data:
        sensors_data = calibrate_sensors_data(sensors_data, sensors)

    # Check paw used or skip wrong paw trials
    if correct_for_paw:
        sensors_data = correct_paw_used(sensors_data, trial.Paw)
    elif trial.Paw.upper() != use_paw:
        continue

    # compute center of gravity
    CoG, centered_CoG = compute_center_of_gravity(sensors_data)

    # Organise all data
    data["name"].append(trial.Video)
    for ch, vals in sensors_data.items():
        data[ch] = vals

    data["CoG"].append(CoG)
    data["centered_CoG"].append(centered_CoG)
    data["start"].append(start_frame)
    data["end"].append(end_frame)

data = pd.DataFrame.from_dict(data)
if not len(data):
    print("No data was loaded, something went wrong!")
else:
    print("Loaded data: ")
    print(data.head())

print("Saving data to: {}".format(savepath))
data.to_hdf(savepath, key='hdf')

