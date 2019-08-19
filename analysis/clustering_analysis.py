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

from analysis.calibrate_sensors import Calibration

# %matplotlib inline

# %%
# Get experiments folders
main_fld = "D:\\Egzona"
sub_flds = {"31":os.path.join(main_fld, "310719"), "13":os.path.join(main_fld, "130819"), "14":os.path.join(main_fld, "140819")}
#"18":os.path.join(main_fld, "180719"), "19":os.path.join(main_fld, "190719")}
#framesfile = os.path.join(main_fld, "clipsframes.csv")

# %%
# Get calibration
calibration = Calibration()

# %%
# Get data
# Load frames times
df = pd.read_csv('D:\Egzona\clipsframes.csv')

# Load data for each video
data = {"name":[], "fr":[], "fl":[], "hr":[], "hl":[], "cg":[], "start":[], "end":[]}
for i, row in df.iterrows():
    fld = sub_flds[row.Video[:2]]
    csv_file, video_files = parse_folder_files(fld, row.Video)
    if csv_file is None: raise ValueError([fld, row.Video])
    else:
        print("Loading file: {}  - -for video: {}".format(csv_file, row.Video))
    sensors_data = load_csv_file(csv_file)

    # Get calibrated sensors data
    sensors = ["fr", "fl", "hr", "hl"]
    calibrated_sensors = {ch:calibration.correct_raw(volts, ch) for ch, volts in sensors_data.items() if ch in sensors}

    # compute center of gravity
    y = (calibrated_sensors["fr"]+calibrated_sensors["fl"]) - (calibrated_sensors["hr"]+calibrated_sensors["hl"])
    x = (calibrated_sensors["fr"]+calibrated_sensors["hr"]) - (calibrated_sensors["fl"]+calibrated_sensors["hl"])

    # Correct for direction of motion and first paw used
    if "b" in row.Direction.lower():
        y = -y
        x = -x
    if "l" in row.Paw.lower():
        x = -x
    else: 
        pass
     
    # Get center of gravity trace
    x,y  = x[row.Start:row.End], y[row.Start:row.End]
    print("Start frame: {} - end frame: {}".format(row.Start, row.End))
    cg = np.vstack([x, y])

    # append to lists
    data["name"].append(row.Video)
    for ch in ["fr", "fl", "hr", "hl"]:
        data[ch].append(sensors_data[ch].values[row.Start:row.End])
    data["cg"].append(cg.T)

    data["start"].append(row.Start)
    data["end"].append(row.End)

data = pd.DataFrame.from_dict(data)
print("Loaded data")
# print(data)

# %%
# Plot stuff
f = plt.figure(figsize=(14, 8))
grid = (5, 6)
cgax = plt.subplot2grid(grid, (0, 0), rowspan=2, colspan=2)

xax = plt.subplot2grid(grid, (0, 2), colspan=2)
yax = plt.subplot2grid(grid, (1, 2), colspan=2)

dxax = plt.subplot2grid(grid, (0, 4), colspan=2)
dyax = plt.subplot2grid(grid, (1, 4), colspan=2)

frax = plt.subplot2grid(grid, (2, 0), colspan=2)
hlax = plt.subplot2grid(grid, (3, 0), colspan=2)
hlfrax = plt.subplot2grid(grid, (2, 2), colspan=2, rowspan=2)

xs, ys, frs, hls = [], [], [], []
for i, row in data.iterrows():
    print(i, row)
    # ? to smooth lines
    #x, y = line_smoother(row.cg[:, 0]-row.cg[0, 0], window_size=11), line_smoother(row.cg[:, 1]-row.cg[0, 1], window_size=11)
    # fr, hl = line_smoother(row.fr[:], window_size=11), line_smoother(row.hl[:], window_size=11)

    x, y = row.cg[:, 0]-row.cg[0, 0], row.cg[:, 1]-row.cg[0, 1]
    fr, hl = row.fr, row.hl

   # Append to lists 
    xs.append(x)
    ys.append(y)
    frs.append(fr)
    hls.append(hl)

    # Plot
    cgax.plot(x, y, lw=1, alpha=.3, color=white)

    xax.plot(x, color=grey, alpha=.25)
    yax.plot(y, color=grey, alpha=.25)

    dxax.plot(np.diff(x), color=grey, alpha=.25)
    dyax.plot(np.diff(y), color=grey, alpha=.25)

    frax.plot(fr, color=grey, alpha=.5)
    hlax.plot(hl, color=grey, alpha=.5)
    hlfrax.plot(hl, fr, color=grey, alpha=.5)

# ! here is wher ethe bug is
try:
    x_mean, y_mean, fr_mean, hr_mean = np.mean(np.vstack(xs), 0), np.mean(np.vstack(ys), 0), np.mean(np.vstack(frs), 0), np.mean(np.vstack(hls), 0)
except:
    # print shapes of different items in lists
    for s,d in zip(["xs", "ys", "frs", "hls"], [xs, ys, frs, hls]):
        print("{} - shape:".format(s))
        print([len(dd) for dd in d])
    raise ValueError("no")
else:
    cgax.plot(x_mean, y_mean, color=red, lw=3, alpha=1)
    xax.plot(x_mean, color=red, lw=6, alpha=1)
    yax.plot(y_mean,  color=red, lw=6, alpha=1)
    dxax.plot(np.diff(x_mean), color=red, lw=6, alpha=1)
    dyax.plot(np.diff(y_mean),  color=red, lw=6, alpha=1)
    hlfrax.plot(fr_mean, hr_mean, color=red, lw=3, alpha=1)

cgax.set(title="center of gravity", xlabel="delta x (g)", ylabel="delta y (g)", xlim=[-10, 15], ylim=[-5, 15])
yax.set(title="y component", xlabel="time", ylabel="delta y (g)", ylim=[-30, 30])
xax.set(title="x component", xlabel="time", ylabel="delta x (g)", ylim=[-30, 30])
dyax.set(title="y component", xlabel="time", ylabel="y speed", ylim=[-5, 5])
dxax.set(title="x component", xlabel="time", ylabel="x speed", ylim=[-5, 5])
frax.set(title="FR", xlabel="time", ylabel="volts", ylim=[0, .5])
hlax.set(title="HL", xlabel="time", ylabel="volts", ylim=[0, .5])
hlfrax.set(title="HL-FR", xlabel="HL", ylabel="FR", )


# %%
f, ax = plt.subplots()
diffs = [hl - fr for hl, fr in zip(hls, frs)]
for fr, hl in zip(frs, hls):
    ax.plot(hl-fr)

#%%
# Get just the CG for the frames of interest
# cgs = [r.cg[r.start:r.end, :]-r.cg[r.start, :] for i,r in data.iterrows()]

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
