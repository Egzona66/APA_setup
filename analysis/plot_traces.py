# %%
import sys
sys.path.append('./')

import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np

from fcutils.file_io.utils import check_file_exists
from fcutils.plotting.colors import salmon
from fcutils.plotting.utils import set_figure_subplots_aspect

# ---------------------------------------------------------------------------- #
#                                     SETUP                                    #
# ---------------------------------------------------------------------------- #

# --------------------------------- Load data -------------------------------- #
main_fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/Egzona/2020"
savepath = os.path.join(main_fld, "data.hdf")
check_file_exists(savepath, raise_error=True)
data = pd.read_hdf(savepath, key='hdf')

sensors = ['fr', 'fl', 'hr', 'hl']

# --------------------------------- Variables -------------------------------- #
calibrated_data = False # Set as true if the data are calibrated Volts -> Grams
plot_centered_CoG = False # if true the centered CoG is used (all trials starts at 0,0)
fps = 600

# %%
# ------------------------------- Create figure ------------------------------ #
f = plt.figure(figsize=(20, 14))

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
        axes[ch].plot(trial[ch], color='k', alpha=.4, lw=2, ls='--', label=label)

    

# --------------------------- Plot sensors means --------------------------- #
means = {ch:data[ch].mean() for ch in sensors}

for ch in sensors:
    axes[ch].plot(means[ch], color=salmon, lw=4, label='median')


# --------------------------------- Plot CoG --------------------------------- #
if plot_centered_CoG:
    CoG = data['centered_CoG']
else:
    CoG = data['CoG']

mean_CoG = CoG.mean()
time = np.arange(CoG[0].shape[0])
axes['CoG'].scatter(mean_CoG[:, 0], mean_CoG[:, 1], c=time, 
                alpha=1, cmap="Reds")



# # -------------------------------- Style plots ------------------------------- #
for ch in sensors:
    axes[ch].legend()

    if calibrated_data:
        ylabel = '$g$'
    else:
        ylabel = '$V$'
    axes[ch].set(ylabel=ylabel)

axes['CoG'].set(ylabel=ylabel, xlabel=ylabel)


# %%

# %%
