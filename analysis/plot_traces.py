# %%
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np
from myterial import light_blue_light, purple_light, lime_light, salmon, white, green, green_dark, salmon_dark

from fcutils.file_io.utils import check_file_exists
from fcutils.file_io.io import load_json
from fcutils.plotting.colors import salmon
from fcutils.plotting.utils import set_figure_subplots_aspect
from fcutils.plotting.plot_elements import plot_line_outlined, plot_mean_and_error
from fcutils.maths.utils import derivative
from loguru import logger

import sys
import os

sys.path.append('./')
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from analysis.utils.utils import get_onset_offset

# ---------------------------------------------------------------------------- #
#                                     SETUP                                    #
# ---------------------------------------------------------------------------- #
plot_trials_CoG = True # if true a line for each CoG is plotted, otherwise just the mean
plot_centered_CoG = True # if true the centered CoG is used (all trials starts at 0,0)

# --------------------------------- Load data -------------------------------- #
main_fld = "D:\\Egzona\\Forceplate\\2021"
data_file = os.path.join(main_fld, "data.hdf")
metadata_file = os.path.join(main_fld, "metadata.json")
check_file_exists(data_file, raise_error=True)
check_file_exists(metadata_file, raise_error=True) 

data = pd.read_hdf(data_file, key='hdf')
metadata = load_json(metadata_file)

sensors = metadata['sensors']
fps = metadata['fps']


logger.info(f'Found {len(data)} trials')



# %%
# ---------------------------------------------------------------------------- #
#                                     extract APA                              #
# ---------------------------------------------------------------------------- #

'''
        For each trial:
                1) check the mouse has been on the sensors for 400ms
                2) check > 90% of mouse weight on sensors
                3) extract precise APA timing (when leading PAW comes off)
'''

liftoff_th = 10
colors = [light_blue_light, purple_light, lime_light, salmon]

good_trials = []
for i, trial in data.iterrows():
        sensors_arrays = np.vstack(trial[sensors].values)
        tot_weight = sensors_arrays.sum(axis=0)

        # get liftoff
        ons, offs = get_onset_offset(trial['fr'], liftoff_th)

        # if len(offs) != 1:
        #         raise ValueError(f'Found weird offs: {offs}')

        # quality controls
        good = True
        if not len(offs):
                logger.warning(f'Did not find paw liftoffs for {trial["name"]} with threshld: {liftoff_th}')
                good = False

        first_off = offs[0]
        if np.median(tot_weight[:first_off]) < 80:
                logger.warning(f'{trial["name"]} less than 80% of weight on sensors before lift off at {first_off}')
                good = False

        if np.any(np.nanmean(sensors_arrays[:, :first_off], 1) < 6):
                logger.warning(f'{trial["name"]} less than 6% of weight on one of the sensors before lift off at {first_off}')
                good = False
        trial['liftoff'] = first_off

        if good:
                good_trials.append(trial)

        # make figure
        f, ax = plt.subplots(figsize=(16, 9))
        f.suptitle(trial['name'] + f' good: {good}')
        xticks = np.arange(0, len(trial['fr']), fps/10)
        xlabels = [round((x/fps), 3) for x in xticks]
        ax.set(xticks=xticks, xlabel='seconds', xticklabels=xlabels)
        ax.axhline(0, lw=2, color='k')
        
        # plot offsets                
        ax.axhline(liftoff_th, color=[.2, .2, .2], lw=2, ls='--', alpha=.5, zorder=-1)
        for off in offs:
                ax.axvline(off)

        # plot tot weight
        ax.plot(tot_weight, lw=2, color='k', label='tot. weight %')

        # plot each paw
        for ch, color in zip(sensors, colors):
                if ch == 'fr':
                        lw = 3
                else:
                        lw = 1
                ax.plot(trial[ch], lw=lw, label=ch)



        ax.legend()


# %%
'''
        Plot traces aligned to lift off
'''
f, axarr = plt.subplots(figsize=(18, 12), nrows=2, sharex=True)
# for trial in good_trials:
#         ax.plot(trial.fr)


for paw, (c1, c2) in zip(('fr', 'fl'), ((green, green_dark), (salmon, salmon_dark))):
        all_trials = np.vstack([t[paw] for t in good_trials])
        axarr[0].plot(all_trials.T, color=c1)
        plot_mean_and_error(np.mean(all_trials, 0), np.std(all_trials, 0), axarr[0], color=c2, label=paw)

        aligned = np.zeros((len(good_trials), 5000))
        liftoffs = []
        for n, trial in enumerate(good_trials):
                # x = np.arange(len(trial[paw])) - trial.liftoff
                # axarr[1].plot(x, trial[paw])

                shift = 1000 - trial.liftoff
                aligned[n, shift:shift+len(trial[paw])] = trial[paw]
                liftoffs.append(trial.liftoff)
                
        aligned = aligned[:, np.min(liftoffs) + 200:]



        axarr[1].plot(aligned.T, color=c1)
        plot_mean_and_error(np.mean(aligned, 0), np.std(aligned, 0), axarr[1], color=c2, label=paw)

axarr[0].legend()
axarr[1].legend()
axarr[1].axhline(liftoff_th, lw=3, color='b')
axarr[0].set(xlim=[0, all_trials.shape[1]], ylabel='weight %')
axarr[1].set(xlabel='Frames', ylabel='weight %')


# %%
'''
        Plot good sensors but normalizes
'''
from sklearn.preprocessing import StandardScaler

f, ax = plt.subplots(figsize=(18, 12), sharex=True)

for paw, (c1, c2) in zip(('fr', 'fl'), ((green, green_dark), (salmon, salmon_dark))):
        normalized = []
        for trial in good_trials:
                baseline = trial[paw][:trial.liftoff]
                normalized.append((trial[paw] - np.mean(baseline)) / np.max(baseline))

        normed = np.vstack(normalized).T
        ax.plot(normed, color=c1)
        plot_mean_and_error(np.mean(normed, 1), np.std(normed, 1), ax, color=c2, label=paw)
ax.legend()
ax.set(xlabel='Frames', ylabel='weight %', xlim=[0, 600])


# %%
# ------------------------------- Create figure ------------------------------ #

f, ax = plt.subplots(figsize=(18, 12))

xticks = np.arange(0, fps, fps/10)
xlabels = [round((x/fps), 3) for x in xticks]
ax.set(xticks=xticks, xlabel='seconds', xticklabels=xlabels)
ax.set( ylim=(0,100))


# --------------------------- Plot sensors means --------------------------- #
means = {ch:data[ch].mean() for ch in sensors}
std = {ch:np.std(data[ch].values) for ch in sensors}


colors = [light_blue_light, purple_light, white, white]

for ch, color in zip(sensors, colors):
        for n, trace in data[ch].iteritems():
                ax.plot(trace, color='red')
                break
#     X= np.arange(len(means[ch]))
#     plot_line_outlined(ax, rolling_mean(means[ch], 3), color=color, lw=4, 
#             outline=0,label=ch, zorder=99)
#     ax.fill_between(X, 
#                          means[ch]-std[ch], means[ch]+std[ch], 
#                          color=color, alpha=.2, zorder=90)
ax.axhline(0, lw=2, color=[.2, .2, .2])
ax.legend(fontsize='xx-large')
f.tight_layout()

plt.show()

# %%
# # ----------------------- Create figure ------------------------------ #
# f = plt.figure(figsize=(18, 12))

# grid = (5, 7)
# axes = {}
# axes['CoG'] = plt.subplot2grid(grid, (1, 0), rowspan=2, colspan=3)
# axes['fr'] = plt.subplot2grid(grid, (1, 4), colspan=3)
# axes['fl'] = plt.subplot2grid(grid, (2, 4), colspan=3, sharey=axes['fr'])
# axes['hr'] = plt.subplot2grid(grid, (3, 4),  colspan=3, sharey=axes['fr']) 
# axes['hl'] = plt.subplot2grid(grid, (4, 4),  colspan=3, sharey=axes['fr'])


# # Style axes
# for ch in ['fr', 'fl', 'hr']:
#     axes[ch].set(xticks=[])

# xticks = np.arange(0, fps, fps/10)
# xlabels = [round((x/fps), 3) for x in xticks]
# axes['hl'].set(xlabel='seconds', xticklabels=xlabels, xticks=xticks)

# sns.despine(offset=10)
# for title, ax in axes.items():
#     ax.set(title=title.upper())

# # -------------------------- Plot individual trials -------------------------- #
# for trn, trial in data.iterrows():
#   for ch in sensors:
#        if trn == 0:
#            label='trials'
#        else:
#            label=None
#        axes[ch].plot(trial[ch], color='k', alpha=.15, lw=2, ls='--', label=label)

    

# # --------------------------- Plot sensors means --------------------------- #
# means = {ch:data[ch].mean() for ch in sensors}
# std = {ch:np.std(data[ch].values) for ch in sensors}

# for ch in sensors:
#     axes[ch].plot(means[ch], color=salmon, lw=4, label='mean', zorder=99)
#     axes[ch].fill_between(np.arange(len(means[ch])), 
#                         means[ch]-std[ch], means[ch]+std[ch], 
#                         color=salmon, alpha=.3, zorder=90)


# # --------------------------------- Plot CoG --------------------------------- #
# if plot_centered_CoG:
#     CoG = data['centered_CoG']
# else:
#     CoG = data['CoG']

# if plot_trials_CoG:
#     axes['CoG'].plot(*CoG.values, color='k', alpha=.15, lw=2, ls='--')

# mean_CoG = CoG.mean()
# time = np.arange(CoG[0].shape[0])
# axes['CoG'].scatter(mean_CoG[:, 0], mean_CoG[:, 1], c=time, 
#                 alpha=1, cmap="Reds", zorder=99)



# # # -------------------------------- Style plots ------------------------------- #
# for ch in sensors:
#     axes[ch].legend()

#     if metadata['calibrate_sensors']:
#         if not metadata['weight_percent']:
#             ylabel = '$g$'
#             axes[ch].set(ylabel=ylabel, ylim=(0,15))
#         else:
#             ylabel = '$weight percent.$'
#             axes[ch].set(ylabel=ylabel, ylim=(0,50))
#     else:
#         ylabel = '$V$'

# axes['CoG'].set(ylabel=ylabel, ylim=(-50, 50), xlabel=ylabel, xlim=(-50, 50))

# f.tight_layout()

# plt.show()