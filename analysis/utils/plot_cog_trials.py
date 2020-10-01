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
from fcutils.plotting.utils import set_figure_subplots_aspect, create_figure

# ---------------------------------------------------------------------------- #
#                                     SETUP                                    #
# ---------------------------------------------------------------------------- #
plot_trials_CoG = True # if true a line for each CoG is plotted, otherwise just the mean
plot_centered_CoG = True # if true the centered CoG is used (all trials starts at 0,0)

# --------------------------------- Load data -------------------------------- #
main_fld = "D:\\Egzona\\2020"
data_file = os.path.join(main_fld, "data.hdf")
metadata_file = os.path.join(main_fld, "metadata.json")
check_file_exists(data_file, raise_error=True)
check_file_exists(metadata_file, raise_error=True)

data = pd.read_hdf(data_file, key='hdf')
metadata = load_json(metadata_file)

sensors = metadata['sensors']
fps = metadata['fps']

# %%
f, axarr =  create_figure(subplots=True, ncols=6, nrows=int((len(data)/6)+1), sharex=True, sharey=True)


for ax, (i, trial) in zip(axarr, data.iterrows()): 
    CoG = trial.centered_CoG
    time = np.arange(CoG.shape[0])
    ax.scatter(CoG[:, 0], CoG[:, 1], c=time, 
                    alpha=1, cmap="Reds", zorder=99)
    ax.plot(CoG[:, 0], CoG[:, 1], color='k', lw=.5, alpha=.4)
    ax.set(title=trial['name'])


f.tight_layout()
