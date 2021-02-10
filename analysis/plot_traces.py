# %%
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from pathlib import Path
import numpy as np
from rich.progress import track
from myterial import light_blue, orange_dark, indigo, salmon, white, green, green_dark, salmon_dark, blue_grey_dark
from scipy.signal import iirnotch, filtfilt, resample

from fcutils.file_io.utils import check_file_exists
from fcutils.file_io.io import load_json
from fcutils.plotting.colors import salmon
from fcutils.plotting.utils import set_figure_subplots_aspect, save_figure, clean_axes
from fcutils.plotting.plot_elements import plot_line_outlined, plot_mean_and_error
from fcutils.maths.utils import derivative, rolling_mean
from loguru import logger

import sys
import os

sys.path.append('./')
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from analysis.utils.utils import get_onset_offset

from analysis.load_data import savepath, metadata_savepath, sensors

# ---------------------------------------------------------------------------- #
#                                     SETUP                                    #
# ---------------------------------------------------------------------------- #
plot_trials_CoG = True # if true a line for each CoG is plotted, otherwise just the mean
plot_centered_CoG = True # if true the centered CoG is used (all trials starts at 0,0)

# --------------------------------- Load data -------------------------------- #
main_fld = "D:\\Egzona\\Forceplate\\2021"
save_fld = Path('D:\\Egzona\\Forceplate\\analysis')

data = pd.read_hdf(savepath, key='hdf')
metadata = load_json(metadata_savepath)

sensors = metadata['sensors']
fps = metadata['fps']

logger.info(f'Loading trial data from {savepath}')
logger.info(f'Found {len(data)} trials')


COLORS = dict(
        fr = light_blue,
        fl = orange_dark,
        hr = indigo,
        hl = salmon,
        tot_weight = blue_grey_dark
)

# %%
# ---------------------------------------------------------------------------- #
#                                     extract APA                              #
# ---------------------------------------------------------------------------- #

'''
        Plot each trial
'''
liftoff_th = 6  # paw < 6% of boy weight is liftoff


xkwargs = dict(
        xticks=[-1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5],
        xticklabels=[-1.5, -1, -.5, -0.25, 0, 0.25, 0.5, 1, 1.5],
)

cog_fig, cog_ax = plt.subplots(figsize=(12, 12))
main_fig, main_axes = plt.subplots(figsize=(18, 12), ncols=2, nrows=2, sharex=True, sharey=True)
main_axes = main_axes.flatten()
clean_axes(main_fig)

# style main figure
for ch, ax in zip(sensors, main_axes):
        ax.set(title=ch, ylabel='weight %', xlabel='time wrt RF take off (s)', xlim=[-.5, .5], **xkwargs)
        ax.axvline(0, lw=4, ls='--', color=[.2, .2, .2], zorder=-1)
        ax.axhline(0, lw=2, color='k', zorder=-1)

step_start_frames, all_sigs = [], []
for i, trial in track(data.iterrows(), total=len(data)):
        start = np.where(trial.fr_on_sensor == 0)[0][0]
        step_start_frames.append(start)


        fig, axarr = plt.subplots(nrows=2, figsize=(16, 9), sharex=True)
        SIGS = []
        for n, ch in enumerate(sensors):
                color = COLORS[ch]

                # force fps to 600
                raw = trial[ch]
                on_sensor = trial[f'{ch}_on_sensor']
                time = np.linspace(-start/trial.fps, (len(raw)-start)/trial.fps, len(raw))

                # notch filter noise
                b_notch, a_notch = iirnotch(2.2, 3, trial.fps)
                sig = filtfilt(b_notch, a_notch, raw)
                SIGS.append(sig)

                # plot raw and altered signals
                lw = 4 if ch not in ('fr') else 5

                axarr[0].plot(
                        time, raw, lw=lw-3, color=color)
                axarr[0].plot(
                        time, sig, lw=lw, color=color, label=ch)

                # plot channels derivatives
                if 'tot' not in ch:
                        axarr[1].plot(
                                time,
                                rolling_mean(derivative(sig), 30), 
                                lw=3, color=color, label=ch)

                # plot also in the main figure
                if 'tot' not in ch:
                        main_axes[n].plot(time[:start], sig[:start], lw=2, color=color, alpha=1)
                        main_axes[n].plot(time[start:], sig[start:], lw=2, color=color, alpha=.25)

                # mark when paw on sensors
                axarr[0].plot(
                        time,
                        120 + 3 * on_sensor - n*5,
                        lw=2, color=COLORS[ch], label=ch)
        all_sigs.append(np.vstack(SIGS).T)

        # plot center of gravity
        cog_ax.plot(
                trial.centered_CoG[:1200, 0],
                trial.centered_CoG[:1200, 1],
                lw=2,
                color=[.2, .2, .2]
        )
        cog_ax.scatter(
                trial.centered_CoG[0, 0],
                trial.centered_CoG[0, 1],
                lw=2,
                edgecolors=[.2, .2, .2],
                color='salmon',
                s=300,
        )
        # mark stuff
        axarr[0].axvline(0, lw=2, color='k', alpha=.6, ls='--', zorder=-1)
        axarr[1].axvline(0, lw=2, color='k', alpha=.6, ls='--', zorder=-1)

        axarr[0].axhline(liftoff_th, lw=6, color='k', alpha=.2, ls='--')
        axarr[0].axhline(0, lw=2, color='k', zorder=-1)
        axarr[0].legend()     
        axarr[1].legend()

        axarr[0].set(ylabel='$W_{pc}$')     

        axarr[1].set(ylabel='$\\frac{d W_{pc}}{dt}$', **xkwargs, xlabel='time (ms)', xlim=[-2, 2])

        # save figure
        clean_axes(fig)
        save_figure(fig, str(save_fld/f"trial_plot_{trial['name']}"), verbose=False)

        if i > 0:
                plt.close(fig)

        # if i == 2:
        #         break   

save_figure(main_fig, str(save_fld/f"all_trials"), verbose=False)
if len(step_start_frames) == len(data):
        data['step_start_frames'] = step_start_frames


# %%
from sklearn.decomposition import PCA
from einops import rearrange

# stack all signals
n = len(all_sigs)
lengths = [s.shape[0] for s in all_sigs]
l = np.max(lengths)

X = np.zeros((n, l, 4))
for num, sig in enumerate(all_sigs):
        X[num, :len(sig), :] = sig

# plot PCA
flatX = repeat(X, 'n l k -> (n l) k')
try:
        pca = PCA(n_components=2).fit(flatX)
except ValueError:
        pca = PCA(n_components=2).fit(flatX)

f, axarr = plt.subplots(ncols=2, figsize=(18, 12))
for trialn in range(n):
        trial = X[trialn, :1200, :]
        pc = pca.transform(trial)
        axarr[0].plot(pc[:, 0], pc[:, 1], color=[.2, .2 , .2], lw=2)
        axarr[0].scatter(pc[0, 0], pc[0, 1], color='salmon', s=100, lw=1, edgecolors=[.2, .2, .2], zorder=100)


# %%
# '''
#         Plot traces aligned to lift off
# '''
# f, axarr = plt.subplots(figsize=(18, 12), nrows=2, sharex=True)
# # for trial in good_trials:
# #         ax.plot(trial.fr)


# for paw, (c1, c2) in zip(('fr', 'fl'), ((green, green_dark), (salmon, salmon_dark))):
#         all_trials = np.vstack([t[paw] for t in good_trials])
#         axarr[0].plot(all_trials.T, color=c1)
#         plot_mean_and_error(np.mean(all_trials, 0), np.std(all_trials, 0), axarr[0], color=c2, label=paw)

#         aligned = np.zeros((len(good_trials), 5000))
#         liftoffs = []
#         for n, trial in enumerate(good_trials):
#                 # x = np.arange(len(trial[paw])) - trial.liftoff
#                 # axarr[1].plot(x, trial[paw])

#                 shift = 1000 - trial.liftoff
#                 aligned[n, shift:shift+len(trial[paw])] = trial[paw]
#                 liftoffs.append(trial.liftoff)
                
#         aligned = aligned[:, np.min(liftoffs) + 200:]



#         axarr[1].plot(aligned.T, color=c1)
#         plot_mean_and_error(np.mean(aligned, 0), np.std(aligned, 0), axarr[1], color=c2, label=paw)

# axarr[0].legend()
# axarr[1].legend()
# axarr[1].axhline(liftoff_th, lw=3, color='b')
# axarr[0].set(xlim=[0, all_trials.shape[1]], ylabel='weight %')
# axarr[1].set(xlabel='Frames', ylabel='weight %')


# # %%
# '''
#         Plot good sensors but normalizes
# '''
# from sklearn.preprocessing import StandardScaler

# f, ax = plt.subplots(figsize=(18, 12), sharex=True)

# for paw, (c1, c2) in zip(('fr', 'fl'), ((green, green_dark), (salmon, salmon_dark))):
#         normalized = []
#         for trial in good_trials:
#                 baseline = trial[paw][:trial.liftoff]
#                 normalized.append((trial[paw] - np.mean(baseline)) / np.max(baseline))

#         normed = np.vstack(normalized).T
#         ax.plot(normed, color=c1)
#         plot_mean_and_error(np.mean(normed, 1), np.std(normed, 1), ax, color=c2, label=paw)
# ax.legend()
# ax.set(xlabel='Frames', ylabel='weight %', xlim=[0, 600])


# # %%
# # ------------------------------- Create figure ------------------------------ #

# f, ax = plt.subplots(figsize=(18, 12))

# xticks = np.arange(0, fps, fps/10)
# xlabels = [round((x/fps), 3) for x in xticks]
# ax.set(xticks=xticks, xlabel='seconds', xticklabels=xlabels)
# ax.set( ylim=(0,100))


# # --------------------------- Plot sensors means --------------------------- #
# means = {ch:data[ch].mean() for ch in sensors}
# std = {ch:np.std(data[ch].values) for ch in sensors}


# colors = [light_blue, orange_dark, white, white]

# for ch, color in zip(sensors, colors):
#         for n, trace in data[ch].iteritems():
#                 ax.plot(trace, color='red')
#                 break
# #     X= np.arange(len(means[ch]))
# #     plot_line_outlined(ax, rolling_mean(means[ch], 3), color=color, lw=4, 
# #             outline=0,label=ch, zorder=99)
# #     ax.fill_between(X, 
# #                          means[ch]-std[ch], means[ch]+std[ch], 
# #                          color=color, alpha=.2, zorder=90)
# ax.axhline(0, lw=2, color=[.2, .2, .2])
# ax.legend(fontsize='xx-large')
# f.tight_layout()

# plt.show()

# # %%
# # # ----------------------- Create figure ------------------------------ #
# # f = plt.figure(figsize=(18, 12))

# # grid = (5, 7)
# # axes = {}
# # axes['CoG'] = plt.subplot2grid(grid, (1, 0), rowspan=2, colspan=3)
# # axes['fr'] = plt.subplot2grid(grid, (1, 4), colspan=3)
# # axes['fl'] = plt.subplot2grid(grid, (2, 4), colspan=3, sharey=axes['fr'])
# # axes['hr'] = plt.subplot2grid(grid, (3, 4),  colspan=3, sharey=axes['fr']) 
# # axes['hl'] = plt.subplot2grid(grid, (4, 4),  colspan=3, sharey=axes['fr'])


# # # Style axes
# # for ch in ['fr', 'fl', 'hr']:
# #     axes[ch].set(xticks=[])

# # xticks = np.arange(0, fps, fps/10)
# # xlabels = [round((x/fps), 3) for x in xticks]
# # axes['hl'].set(xlabel='seconds', xticklabels=xlabels, xticks=xticks)

# # sns.despine(offset=10)
# # for title, ax in axes.items():
# #     ax.set(title=title.upper())

# # # -------------------------- Plot individual trials -------------------------- #
# # for trn, trial in data.iterrows():
# #   for ch in sensors:
# #        if trn == 0:
# #            label='trials'
# #        else:
# #            label=None
# #        axes[ch].plot(trial[ch], color='k', alpha=.15, lw=2, ls='--', label=label)

    

# # # --------------------------- Plot sensors means --------------------------- #
# # means = {ch:data[ch].mean() for ch in sensors}
# # std = {ch:np.std(data[ch].values) for ch in sensors}

# # for ch in sensors:
# #     axes[ch].plot(means[ch], color=salmon, lw=4, label='mean', zorder=99)
# #     axes[ch].fill_between(np.arange(len(means[ch])), 
# #                         means[ch]-std[ch], means[ch]+std[ch], 
# #                         color=salmon, alpha=.3, zorder=90)


# # # --------------------------------- Plot CoG --------------------------------- #
# # if plot_centered_CoG:
# #     CoG = data['centered_CoG']
# # else:
# #     CoG = data['CoG']

# # if plot_trials_CoG:
# #     axes['CoG'].plot(*CoG.values, color='k', alpha=.15, lw=2, ls='--')

# # mean_CoG = CoG.mean()
# # time = np.arange(CoG[0].shape[0])
# # axes['CoG'].scatter(mean_CoG[:, 0], mean_CoG[:, 1], c=time, 
# #                 alpha=1, cmap="Reds", zorder=99)



# # # # -------------------------------- Style plots ------------------------------- #
# # for ch in sensors:
# #     axes[ch].legend()

# #     if metadata['calibrate_sensors']:
# #         if not metadata['weight_percent']:
# #             ylabel = '$g$'
# #             axes[ch].set(ylabel=ylabel, ylim=(0,15))
# #         else:
# #             ylabel = '$weight percent.$'
# #             axes[ch].set(ylabel=ylabel, ylim=(0,50))
# #     else:
# #         ylabel = '$V$'

# # axes['CoG'].set(ylabel=ylabel, ylim=(-50, 50), xlabel=ylabel, xlim=(-50, 50))

# # f.tight_layout()

# # plt.show()