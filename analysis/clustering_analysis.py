# %%
import sys
sys.path.append("./")

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial import distance


from utils.video_utils import Editor as VideoUtils
from forceplate_config import Config
from utils.maths.filtering import line_smoother
from utils.maths.math_utils import get_n_colors


from utils.file_io_utils import *
from utils.analysis_utils import *
from utils.video_utils import Editor as VideoUtils
from utils.matplotlib_config import *
from utils.plotting_utils import *
from utils.constants import *

from calibrate_sensors import Calibration

# %matplotlib inline

# %%
# Get experiments folders
main_fld = "D:\\Egzona\\2020"
sub_flds = {"21":os.path.join(main_fld, "21012020"),"23":os.path.join(main_fld, "23012020"), "24":os.path.join(main_fld, "24012020"), "28":os.path.join(main_fld, "28012020"), "29":os.path.join(main_fld, "29012020")} 
#"18":os.path.join(main_fld, "180719"), "19":os.path.join(main_fld, "190719") 
framesfile = os.path.join(main_fld, "clipsframes_FP3.csv")

# %%
# Get calibration
calibration = Calibration()

# %%
# Get data
# Load frames times
df = pd.read_csv('D:\Egzona\clipsframes_FP3.csv')


# ! important params
target_fps = 600

# Load data for each video
data = {"name":[], "fr":[], "fl":[], "hr":[], "hl":[], "cg":[], "start":[], "end":[]}
for i, row in df.iterrows():
    try:
        fld = sub_flds[row.Video[:2]]
    except Exception as e:  
        raise ValueError("Could not find folder! {}".format(e))

    csv_file, video_files = parse_folder_files(fld, row.Video)
    if csv_file is None: 
        raise ValueError("cvs file is None")
    else:
        print("Loading file: {}  - -for video: {}".format(csv_file, row.Video))
    sensors_data = load_csv_file(csv_file)

    # Get baselined and calibrated sensors data
    sensors = ["fr", "fl", "hr", "hl"]
    # calibrated_sensors = {ch:calibration.correct_raw(baseline_sensor_data(volts), ch) 
    #                             for ch, volts in sensors_data.items() if ch in sensors}
    calibrated_sensors = sensors_data
 
    # ? Get resampled data (from frames to ms)
    fps = 600 # int(row["Frame rate"])
    secs = len(calibrated_sensors["fr"])/fps
    n_samples = int(secs * target_fps)
    calibrated_sensors = {ch:upsample_timeseries(data, n_samples) for ch, data in calibrated_sensors.items()}

    # compute center of gravity
    y = (calibrated_sensors["fr"]+calibrated_sensors["fl"]) - (calibrated_sensors["hr"]+calibrated_sensors["hl"])
    x = (calibrated_sensors["fr"]+calibrated_sensors["hr"]) - (calibrated_sensors["fl"]+calibrated_sensors["hl"])

    # Correct for direction of motion and first paw used
    # if "l" in row.Paw.lower():
    #     x = -x

     
    # Get center of gravity trace
    end = row.End 
    start= int(row.Start)
    x,y  = x[start:end], y[start:end]
    print("Start frame: {} - end frame: {} - {}  nframes".format(start, end, (end - start)))
    cg = np.vstack([x, y])

    # append to lists
    data["name"].append(row.Video)
    for ch in ["fr", "fl", "hr", "hl"]:
        if len(calibrated_sensors[ch][start:end]) != (end - start):
            raise ValueError(ch)
        data[ch].append(calibrated_sensors[ch][start:end])
    data["cg"].append(cg.T)
    data["start"].append(start)
    data["end"].append(end)

data = pd.DataFrame.from_dict(data)
print("Loaded data")
print(data)

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
# %%
f, ax = plt.subplots()
diffs = [hl - fr for hl, fr in zip(hls, frs)]
for fr, hl in zip(frs, hls):
    ax.plot(hl-fr)

#%%
# Get just the CG for the frames of interest
cgs = [r.cg[r.start:r.end, :]-r.cg[r.start, :] for i,r in data.iterrows()]

# ? Compute distances matrixes
dist_matrix = np.zeros((len(diffs), len(diffs)))
for i in np.arange(len(diffs)):
    for ii in np.arange(len(diffs)):
        #d = np.sum(calc_distance_between_points_two_vectors_2d(diffs[i], diffs[ii]))
        d = np.sum(np.abs(diffs[i]-diffs[ii]))
        dist_matrix[i, ii] = d

f, ax = plt.subplots()
ax.imshow(dist_matrix)
ax.set(title="distance matrix", xlabel="trial #", ylabel="trial #")

# %%
# Clustering
Z = linkage(distance.squareform(dist_matrix))

#%%
f, ax = plt.subplots()
dendo = dendrogram(Z)

max_d = 2
ax.axhline(max_d, color=grey, lw=2, ls="--")

clust = fcluster(Z, max_d, criterion="distance")
print(len(clust), clust)


ax.set(title="dendogram", xlabel="trial idx", ylabel="distance")


#%%
f, ax = plt.subplots()

colors = get_n_colors(max(clust)+1)
for d, c in zip(diffs, clust):
    ax.plot(d, color=colors[c])

#%%
