# %%
import sys
sys.path.append('./')

import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np

from fcutils.file_io.utils import check_file_exists
from fcutils.file_io.io import load_json
from fcutils.plotting.colors import salmon
from fcutils.plotting.utils import set_figure_subplots_aspect

# ---------------------------------------------------------------------------- #
#                                     SETUP                                    #
# ---------------------------------------------------------------------------- #
plot_trials_CoG = True # if true a line for each CoG is plotted, otherwise just the mean
plot_centered_CoG = True # if true the centered CoG is used (all trials starts at 0,0)

# --------------------------------- Load data -------------------------------- #
main_fld = "D:\\Egzona\\Forceplate\\2020"
data_file = os.path.join(main_fld, "data.hdf")
metadata_file = os.path.join(main_fld, "metadata.json")
check_file_exists(data_file, raise_error=True)
check_file_exists(metadata_file, raise_error=True) 

data = pd.read_hdf(data_file, key='hdf')
metadata = load_json(metadata_file)

sensors = metadata['sensors']
fps = metadata['fps']

# %%
# ------------------------------- Create figure ------------------------------ #
f = plt.figure(figsize=(18, 12))

grid = (5, 7)
axes = {}
axes['CoG'] = plt.subplot2grid(grid, (1, 0), rowspan=2, colspan=3)
axes['fr'] = plt.subplot2grid(grid, (0, 4), colspan=3)
axes['fl'] = plt.subplot2grid(grid, (1, 4), colspan=3, sharey=axes['fr'])
axes['hr'] = plt.subplot2grid(grid, (2, 4),  colspan=3, sharey=axes['fr'])
axes['hl'] = plt.subplot2grid(grid, (3, 4),  colspan=3, sharey=axes['fr'])

# Style axes
for ch in ['fr', 'fl', 'hr']:
    axes[ch].set(xticks=[])

xticks = np.arange(0, fps, fps/10)
xlabels = [round((x/fps), 3) for x in xticks]
axes['hl'].set(xlabel='seconds', xticklabels=xlabels, xticks=xticks)

sns.despine(offset=10)
for title, ax in axes.items():
    ax.set(title=title.upper())

# -------------------------- Plot individual trials -------------------------- #
for trn, trial in data.iterrows():
    for ch in sensors:
        if trn == 0:
            label='trials'
        else:
            label=None
        axes[ch].plot(trial[ch], color='k', alpha=.15, lw=2, ls='--', label=label)

    

# --------------------------- Plot sensors means --------------------------- #
means = {ch:data[ch].mean() for ch in sensors}
std = {ch:np.std(data[ch].values) for ch in sensors}

for ch in sensors:
    axes[ch].plot(means[ch], color=salmon, lw=4, label='mean', zorder=99)
    axes[ch].fill_between(np.arange(len(means[ch])), 
                        means[ch]-std[ch], means[ch]+std[ch], 
                        color=salmon, alpha=.3, zorder=90)


# --------------------------------- Plot CoG --------------------------------- #
if plot_centered_CoG:
    CoG = data['centered_CoG']
else:
    CoG = data['CoG']

if plot_trials_CoG:
    axes['CoG'].plot(*CoG.values, color='k', alpha=.15, lw=2, ls='--')

mean_CoG = CoG.mean()
time = np.arange(CoG[0].shape[0])
axes['CoG'].scatter(mean_CoG[:, 0], mean_CoG[:, 1], c=time, 
                alpha=1, cmap="Reds", zorder=99)



# # -------------------------------- Style plots ------------------------------- #
for ch in sensors:
    axes[ch].legend()

    if metadata['calibrate_sensors']:
        if not metadata['weight_percent']:
            ylabel = '$g$'
            axes[ch].set(ylabel=ylabel, ylim=(0,15))
        else:
            ylabel = '$weight percent.$'
            axes[ch].set(ylabel=ylabel, ylim=(0,100))
    else:
        ylabel = '$V$'

axes['CoG'].set(ylabel=ylabel, ylim=(-50, 50), xlabel=ylabel, xlim=(-50, 50))

f.tight_layout()

plt.show()

# %%
