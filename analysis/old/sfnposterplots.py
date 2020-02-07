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


#from utils.video_utils import Editor as VideoUtils
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

import matplotlib.patches as patches

# %matplotlib inline

# %%
# Get experiments folders
main_fld = "D:\\Egzona\\2020"
sub_flds = {"21":os.path.join(main_fld, "21012020")},#"31":os.path.join(main_fld, "310719"), "13":os.path.join(main_fld, "130819")} 
#"18":os.path.join(main_fld, "180719"), "19":os.path.join(main_fld, "190719") 
#framesfile = os.path.join(main_fld, "clipsframes.csv")

# %%
# Get calibration
calibration = Calibration()

# %%
# Get data
# Load frames times
df = pd.read_csv('D:\Egzona\clipsframes.csv')


# ! important params
target_fps = 600
start_shift = 150 # number of milliseconds to take before what is specified in the CSV file

# Load data for each video
data = {"name":[], "fr":[], "fl":[], "hr":[], "hl":[], "cg":[], "start":[], "end":[]}
for i, row in df.iterrows():
    try:
        fld = sub_flds[row.Video[:2]]
    except:
        continue

    csv_file, video_files = parse_folder_files(fld, row.Video)
    if csv_file is None: 
        continue
    else:
        print("Loading file: {}  - for video: {}".format(csv_file, row.Video))
    sensors_data = load_csv_file(csv_file)

    # Get baselined and calibrated sensors data
    sensors = ["fr", "fl", "hr", "hl"]
    calibrated_sensors = {ch:calibration.correct_raw(baseline_sensor_data(volts), ch) for ch, volts in sensors_data.items() if ch in sensors}
 
    # ? Get resampled data (from frames to ms)
    fps = int(row["Frame rate"])
    secs = len(calibrated_sensors["fr"])/fps
    n_samples = int(secs * target_fps)
    calibrated_sensors = {ch:upsample_timeseries(data, n_samples) for ch, data in calibrated_sensors.items()}


    # Correct for direction of motion and first paw used
    if "b" in row.Direction.lower():
        _calibrated_sensors = {}
        _calibrated_sensors['fr'] = calibrated_sensors['hl']
        _calibrated_sensors['hr'] = calibrated_sensors['fl']
        _calibrated_sensors['hl'] = calibrated_sensors['fr']
        _calibrated_sensors['fl'] = calibrated_sensors['hr']
        calibrated_sensors = _calibrated_sensors


    if "l" in row.Paw.lower():
        _calibrated_sensors = {}
        _calibrated_sensors['fr'] = calibrated_sensors['fl']
        _calibrated_sensors['hr'] = calibrated_sensors['hl']
        _calibrated_sensors['hl'] = calibrated_sensors['hr']
        _calibrated_sensors['fl'] = calibrated_sensors['fr']
        calibrated_sensors = _calibrated_sensors
     

    # compute center of gravity
    y = (calibrated_sensors["fr"]+calibrated_sensors["fl"]) - (calibrated_sensors["hr"]+calibrated_sensors["hl"])
    x = (calibrated_sensors["fr"]+calibrated_sensors["hr"]) - (calibrated_sensors["fl"]+calibrated_sensors["hl"])

    # Get center of gravity trace
    start= int(np.floor(row.Start/fps*target_fps)) - start_shift
    end = int(start + target_fps*.6) + start_shift
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
# print(data)

# %%
# Plot stuff
f = plt.figure(figsize=(14, 14), facecolor="white")
grid = (2, 4)


flax = plt.subplot2grid(grid, (0, 0), colspan=2)
hlax = plt.subplot2grid(grid, (1, 0), colspan=2)
frax = plt.subplot2grid(grid, (0, 2), colspan=2)
hrax = plt.subplot2grid(grid, (1, 2), colspan=2)

fls, hrs, frs, hls, xs, ys = [], [], [], [], [], []
excluded = 0
medians_dict = {}
for i, row in data.iterrows():
    x, y = row.cg[:, 0]-row.cg[0, 0], row.cg[:, 1]-row.cg[0, 1]
    fr, hl, fl, hr = row.fr, row.hl, row.fl, row.hr

    # Append to lists if at start of trial each sensor has at least 1g of force applied
    check = True
    for sens, lst in zip([fr, hl, fl, hr], [frs, hls, fls, hrs]):
         if sens[0 + start_shift] < 1 and check: # ! this selects which trials are excluded
             excluded += 1
             check = False
    if not check: continue
    
    for i, (sens, lst) in enumerate(zip([fr, hl, fl, hr], [frs, hls, fls, hrs])):
        if np.abs(len(sens) - np.mean([len(x) for x in lst])) != 0 and lst:
            raise ValueError([i, len(sens), [len(x) for x in lst]])
        lst.append(sens)

    xs.append(x)
    ys.append(y)

    # Plot
    trial_color = [.6, .6, .6]
    frax.plot(line_smoother(fr, window_size=15), color=trial_color, alpha=.5)
    hlax.plot(line_smoother(hl, window_size=15), color=trial_color, alpha=.5)
    flax.plot(line_smoother(fl, window_size=15), color=trial_color, alpha=.5)
    hrax.plot(line_smoother(hr, window_size=15), color=trial_color, alpha=.5)

try:
    fr_median, hl_median, fl_median, hr_median = np.median(np.vstack(frs), 0), np.median(np.vstack(hls), 0),  np.median(np.vstack(fls), 0),  np.median(np.vstack(hrs), 0)
    x_median, y_median = np.median(np.vstack(xs), 0), np.median(np.vstack(ys), 0)
except:
    raise ValueError("no")
else:
    time = np.arange(len(fr_median))

    # cool colors yo
    dtime = np.zeros_like(fr_median)
    t0, t1 = np.where(fr_median < 1)[0][0], np.where(hl_median < 1)[0][0]
    dtime[t0:t1] = 1
    dtime[t1:] = 2


    median_color = 'k'
    # frax.plot(time, line_smoother(fr_median, window_size=15), color=[.6, .6, .9], lw=5)
    # hlax.plot(time, line_smoother(hl_median, window_size=15), color=[.6, .9, .6], lw=5)
    # flax.plot(time, line_smoother(fl_median, window_size=15), color=median_color, lw=5)
    # hrax.plot(time, line_smoother(hr_median, window_size=15), color=median_color, lw=5)
    frax.plot(time, fr_median, color=[.6, .6, .9], lw=5)
    hlax.plot(time, hl_median, color=[.6, .9, .6], lw=5)
    flax.plot(time, fl_median, color=median_color, lw=5)
    hrax.plot(time, hr_median, color=median_color, lw=5)
    medians_dict['fr'] = fr_median
    medians_dict['hr'] = hr_median
    medians_dict['fl'] = fl_median
    medians_dict['hl'] = hl_median

print("Excluded {} of {} trials".format(excluded, i+1))


frax.set(title="FR")
hlax.set(title="HL")
flax.set(title="FL")
hrax.set(title="HR")

# frax.axhline(2.8, color=[.2, .2, .2], lw=2, ls='--')
# frax.axhline(.5, color=[.2, .2, .2], lw=2, ls='--')
# hlax.axhline(4.8, color=[.2, .2, .2], lw=2, ls='--')
# hlax.axhline(.6, color=[.2, .2, .2], lw=2, ls='--')

for ch, ax in zip(['fr', 'hl', 'fl', 'hr'], [frax, hlax, flax, hrax]):
    ax.set(xlabel="time", ylabel="(g)", xlim=[100, 150+start_shift], ylim=[0, 25], 
            xticks=np.arange(0, 300+start_shift, 50), xticklabels=np.arange(-start_shift, 300+start_shift+1, 50))


    # if ch in ['hl', 'fr']:
    # FR
    ax.add_patch(patches.Rectangle((start_shift+9, 23), 40, 1,  linewidth=3, color=[.6, .6, .9], facecolor='none', alpha=1, zorder=-1, ))
    ax.add_patch(patches.Rectangle((start_shift+9, 0), 40, 23,  linewidth=3, edgecolor=[.6, .6, .9], facecolor='none', alpha=.8, zorder=-10, ))

    # HL
    ax.add_patch(patches.Rectangle((start_shift+13, 21.5), 49, 1, linewidth=3,  color=[.6, .9, .6], facecolor='none', alpha=1, zorder=-1, ))
    ax.add_patch(patches.Rectangle((start_shift+13, 0), 49, 21.5, linewidth=3,  edgecolor=[.6, .9, .6], facecolor='none', alpha=.8, zorder=-10, ))



medians = pd.DataFrame.from_dict(medians_dict)

# %%
medians.to_csv("APA_medians_for_nunu.csv")

#%%
# extract movement times from median traces

# Get right paw movement
r_start_th, r_end_th = 2.8, .5
l_start_th, l_end_th = 4.8, .6

r_start = np.where(medians['fr'].values <= r_start_th)[0][0] - start_shift
r_end = np.where(medians['fr'].values <= r_end_th)[0][0] - start_shift

l_start = np.where(medians['hl'].values <= l_start_th)[0][0] - start_shift
l_end = np.where(medians['hl'].values <= l_end_th)[0][0] - start_shift

print("Right paw starts at {} ms and ends at {} ms. Movement lasts: {}".format(r_start, r_end, (r_end - r_start)))
print("Left paw starts at {} ms and ends at {} ms. Movement lasts: {}".format(l_start, l_end, (l_end - l_start)))


#%%
# ? DLC PLOTS

datafld = "D:\\Egzona\\Plot-videos DLC\\data"

# Get the video names
vids = os.listdir(datafld)
vid_names = [v.split("Deep")[0] for v in vids if 'top' not in v]

side_vids = [os.path.join(datafld, v) for v in vids if 'top' not in v]
top_vids = [os.path.join(datafld, v) for v in vids if 'top' in v]

# videos metadata#
metadata = pd.read_csv('D:\Egzona\clipsframes.csv')
target_fps = 500

#%%
example_trace = "130819_F1R2a"

f = plt.figure(figsize=(14, 14), facecolor="white")
grid = (2, 1)


sax = plt.subplot2grid(grid, (0, 0))
tax = plt.subplot2grid(grid, (1, 0))

# ! PARAMS
pcutoff = 0.3

colors = {  'back right paw':'blues',
            'back left paw':'greens',
            'front right paw':'reds',
            'front left paw':'oranges',
            'nose':'Purples',

}

cmaps = {  'back right paw':'Blues',
            'back left paw':'Greens',
            'front right paw':'Reds',
            'front left paw':'Oranges',
            'nose':'Purples',

}

sidecolors = {
            # 'nose': '#283A8C',
            'back left paw': 'Blues',
            'back left toes': 'Greens', 
            'front left paw': 'Reds',
            'front left toes': 'YlOrBr',
            'front right toes': 'Reds',
            'front right paw': 'YlOrBr',
            'back right paw': 'Blues',
            'back right toes': 'Greens',
            'neck': 'Purples',
            'back': 'RdPu',
            # 'tail':'#A02221',
            }

sidecmaps = {
            # 'nose': '#283A8C',
            'back left paw': 'Blues',
            'back left toes': 'Greens', 
            'front left paw': 'Reds',
            'front left toes': 'YlOrBr',
            'front right toes': 'Reds',
            'front right paw': 'YlOrBr',
            'back right paw': 'Blues',
            'back right toes': 'Greens',
            'neck':  'Purples',
            'back': 'RdPu',
            # 'tail':'#A02221',
}


frame_center = [240, 160]

example_tracking = {}
for vid in vid_names:
    vidmeta = metadata.loc[metadata.Video == vid].iloc[0]

    if vidmeta.Direction.lower() == "b":
        back = True
    else: 
        back = False
    
    if vidmeta.Paw.lower() == "l":
        left = True
    else: 
        left = False

    # TOP
    top = pd.read_hdf([f for f in top_vids if vid in f][0])
    tscorer = top.columns.get_level_values(0)[0]
    top_bps = set(top[tscorer].columns.get_level_values(0))

    for bp in top_bps:
        if bp not in list(colors.keys()): continue

        Index = top[tscorer][bp]['likelihood'].values > pcutoff
        x, y = top[tscorer][bp]['x'].values[Index], top[tscorer][bp]['y'].values[Index]

        if back:
            deltax, deltay = -(x-frame_center[0]), -(y-frame_center[1])+60
        else:
            deltax, deltay = x-frame_center[0], y-frame_center[1]

        if vid != example_trace:
            col = [.7, .7, .7]
            zorder = 4
            alpha = 1
            tax.scatter(deltax+150, deltay+125, color=col, alpha=alpha, zorder=zorder)
        else:
            col = colors[bp]
            zorder=99
            alpha=1
            tax.scatter(deltax+150, deltay+125, c=np.arange(len(deltax)), cmap=cmaps[bp], alpha=alpha, zorder=zorder)
        

        
    # SIDE
    side = pd.read_hdf([f for f in side_vids if vid in f][0])
    tscorer = side.columns.get_level_values(0)[0]
    side_bps = set(side[tscorer].columns.get_level_values(0))

    for bp in side_bps:
        if bp not in list(sidecolors.keys()): continue

        if back:
            if "left" in bp: continue
        else: 
            if "right" in bp: continue    


        Index = side[tscorer][bp]['likelihood'].values > pcutoff
        x, y = side[tscorer][bp]['x'].values[Index], -side[tscorer][bp]['y'].values[Index]

        if back:
            deltax, deltay = (x-frame_center[0]), y-frame_center[1]
        else:
            deltax, deltay = -(x-frame_center[0]), y-frame_center[1]

        if vid != example_trace:
            col = [.7, .7, .7]
            zorder = 4
            alpha = 1
            sax.scatter(deltax+150, deltay+460, color=col, alpha=alpha, zorder=zorder)

        else:
            col = sidecolors[bp]
            zorder=999
            alpha=1
            example_tracking[bp] = [deltax+150, deltay+460]
            sax.scatter(deltax+150, deltay+460, c=np.arange(len(deltax)), cmap=sidecmaps[bp], alpha=alpha, zorder=zorder)





    # sax.invert_yaxis()
    # tax.invert_yaxis()

    for ax in [sax, tax]:
        ax.set(xlim=[0, 480], ylim=[0, 320])





#%%

skeleton1 = [
    ['neck', 'back'],
    ['back', 'back right paw'],
    ['back right paw', 'back right toes'],
]

skeleton2= [
    ['neck', 'back'],
    ['front right paw', 'front right toes'],
    ['front right paw', 'back']
]

t = np.arange(0, 400, 5)

for frame in t:
    for bp1, bp2 in skeleton1:
        try:
            x1, y1 = example_tracking[bp1][0][frame], example_tracking[bp1][1][frame]
            x2, y2 = example_tracking[bp2][0][frame], example_tracking[bp2][1][frame]

            sax.plot([x1,x2], [y1, y2], color="k", lw=3, alpha=.3, zorder=99)
        except:
            pass
    for bp1, bp2 in skeleton2:
        try:
            x1, y1 = example_tracking[bp1][0][frame], example_tracking[bp1][1][frame]
            x2, y2 = example_tracking[bp2][0][frame], example_tracking[bp2][1][frame]

            sax.plot([x1,x2], [y1, y2], color="m", lw=3, alpha=.3, zorder=99)
        except:
            pass

#%%
