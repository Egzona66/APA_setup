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
    CoG = compute_center_of_gravity(sensors_data)

    # Organise all data
    data["name"].append(trial.Video)
    for ch, vals in sensors_data.items():
        data[ch] = vals

    data["cg"].append(CoG)
    data["start"].append(start_frame)
    data["end"].append(end_frame)

data = pd.DataFrame.from_dict(data)
if not len(data):
    print("No data was loaded, something went wrong!")
else:
    print("Loaded data: ")
    print(data.head())

# %%
# Plot stuff
f = plt.figure(figsize=(14, 14))
grid = (4, 4)
cgax = plt.subplot2grid(grid, (0, 0), rowspan=2, colspan=2)
cgtax = plt.subplot2grid(grid, (0, 2), rowspan=2, colspan=2)

flax = plt.subplot2grid(grid, (2, 0), colspan=2)
hlax = plt.subplot2grid(grid, (3, 0), colspan=2)
frax = plt.subplot2grid(grid, (2, 2), colspan=2)
hrax = plt.subplot2grid(grid, (3, 2), colspan=2)

fls, hrs, frs, hls, xs, ys = [], [], [], [], [], []
excluded = 0
for trn, row in data.iterrows():
    x, y = row.cg[:, 0]-row.cg[0, 0], row.cg[:, 1]-row.cg[0, 1]
    fr, hl, fl, hr = row.fr, row.hl, row.fl, row.hr

    # Append to lists if at start of trial each sensor has at least 1g of force applied
    check = True
    # for sens, lst in zip([fr, hl, fl, hr], [frs, hls, fls, hrs]):
    #      if sens[0] < 1 and check:
    #          excluded += 1
    #          check = False
    # if not check: continue
    
    for i, (sens, lst) in enumerate(zip([fr, hl, fl, hr], [frs, hls, fls, hrs])):
        if np.abs(len(sens) - np.mean([len(x) for x in lst])) != 0 and lst:
            raise ValueError("Something went wrong, probably not all recordings have the same number of frames {}"\
                            .format([i, len(sens), [len(x) for x in lst]]))
        lst.append(sens)

    xs.append(x)
    ys.append(y)

    # Plot
    cgtax.plot(x, y, lw=1, alpha=.3, color=white)
    frax.plot(fr, color=grey, alpha=.5)
    hlax.plot(hl, color=grey, alpha=.5)
    flax.plot(fl, color=grey, alpha=.5)
    hrax.plot(hr, color=grey, alpha=.5)

try:
    fr_median, hl_median, fl_median, hr_median = np.median(np.vstack(frs), 0), np.median(np.vstack(hls), 0),  np.median(np.vstack(fls), 0),  np.median(np.vstack(hrs), 0)
    x_median, y_median = np.median(np.vstack(xs), 0), np.median(np.vstack(ys), 0)
except:
    raise ValueError("no")
    # pass
else:
    time = np.arange(len(fr_median))

    # cool colors yo
    dtime = np.zeros_like(fr_median)
    # t0, t1 = np.where(fr_median < 1)[0][0], np.where(hl_median < 1)[0][0]
    # dtime[t0:t1] = 1
    # dtime[t1:] = 2

    cgax.scatter(x_median, y_median, c=time, alpha=1, zorder=10, cmap="Reds")
    cgtax.scatter(x_median, y_median, c=dtime, alpha=1, cmap="tab20c", zorder=10)
    frax.plot(time, fr_median, color=red, lw=4)
    hlax.plot(time, hl_median, color=red, lw=4)
    flax.plot(time, fl_median, color=red, lw=4)
    hrax.plot(time, hr_median, color=red, lw=4)

print("Excluded {} of {} trials".format(excluded, trn+1))

cgax.set(title="center of gravity", xlabel="delta x (g)", ylabel="delta y (g)", xlim=[-15, 15], ylim=[-5, 20])
cgtax.set(title="center of gravity", xlabel="delta x (g)", ylabel="delta y (g)", xlim=[-15, 15], ylim=[-5, 20])
frax.set(title="FR", xlabel="time", ylabel="(g)")
hlax.set(title="HL", xlabel="time", ylabel="(g)")
flax.set(title="FL", xlabel="time", ylabel="(g)")
hrax.set(title="HR", xlabel="time", ylabel="(g)")
