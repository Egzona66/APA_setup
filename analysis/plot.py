import numpy as np
from pathlib import Path
import sys
import os
import matplotlib.pylplots as plt

sys.path.append("./")
sys.path.append(os.path.abspath(os.path.join("..")))

from fcutils.plot.elements import plot_mean_and_error
from fcutils.progress import track
from fcutils.plot.figure import save_figure
from myterial import teal_dark, green, green_light

from analysis.process_data import DataProcessing
from analysis.fixtures import sensors, colors, dark_colors
from analysis._plot import initialize_trial_figure

"""
    Plots individual trials and population averages.
    Current plots:  
        - CoM traces
        - indidivual sensors traces
"""
save_fld = Path("D:\\Egzona\\Forceplate\\analysis")

# load previously saved data
data = DataProcessing.reload()


# initialize pooled trials figure
f, axes = initialize_trial_figure(
    "trials", data.fps, data.n_secs_before, data.n_secs_after,
)

# ----------------------------- loop over trials ----------------------------- #
for i, trial in track(
    data.data.iterrows(), total=len(data.data), description="Plotting..."
):
    # create a trial-specific figure
    trial_f, trial_axes = initialize_trial_figure(
        trial["name"], data.fps, data.n_secs_before, data.n_secs_after,
    )

    # plot on both trial figure and main figure
    for figure_axes in (axes, trial_axes):
        for sensor in sensors + ["tot_weight"]:
            figure_axes[sensor].plot(
                trial[sensor], lw=2, color=colors[sensor], alpha=0.8
            )

        # plot CoG (and mark start/end)
        axes["Cog"].plot(trial.CoG[:, 0], trial.CoG[:, 1], lw=2, color=colors["CoG"])
        axes["Cog"].scatter(*trial.CoG[0, :], lw=2, color=teal_dark)
        axes["Cog"].scatter(
            *trial.CoG[int(data.n_secs_before * data.fps), :],
            zorder=100,
            lw=2,
            color=green,
            label="movement onset",
        )
        axes["Cog"].scatter(
            *trial.CoG[int(data.n_secs_after * data.fps), :],
            zorder=100,
            lw=2,
            color=green_light,
        )

    # save and close trial figure
    trial_axes["CoG"].legend()
    save_figure(save_fld / f'trial_{trial["name"]}')

# ------------------------------- plt averages ------------------------------- #
for ch in sensors + ["tot_weight", "CoG"]:
    avg = np.mean(np.vstack(data.trials[ch].values), axis=0)
    std = np.mean(np.vstack(data.trials[ch].values), axis=0)

    if ch != "CoG":
        plot_mean_and_error(avg, std, axes[ch], color=dark_colors[ch], lw=5)
    else:
        axes[ch].plot(avg[:, 0], avg[:, 1], lw=4, color=dark_colors[ch])

# save main figure and show
save_figure(save_fld / "all_trials")
plt.show()
