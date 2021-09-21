import numpy as np
import sys
import os
import matplotlib.pyplot as plt


sys.path.append("./")
sys.path.append(os.path.abspath(os.path.join("..")))

from fcutils.plot.elements import plot_mean_and_error
from fcutils.progress import track

from analysis.process_data import DataProcessing
from analysis.fixtures import sensors, colors, dark_colors
from analysis._plot import initialize_trial_figure

"""
    Plots individual trials and population averages.
    Current plots:  
        - CoM traces
        - indidivual sensors traces
"""

# load previously saved data
data = DataProcessing.reload()
save_fld = data.main_fld.parent
save_fld.mkdir(exist_ok=True)

# initialize pooled trials figure6
main_f, main_axes = initialize_trial_figure(
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
    for axes_n, figure_axes in enumerate((main_axes, trial_axes)):
        alpha = 0.4 if axes_n == 0 else 0.8

        for sensor in sensors + ["tot_weight"]:
            figure_axes[sensor].plot(
                trial[sensor], lw=2, color=colors[sensor], alpha=alpha, label=sensor,
            )

        # plot CoG (and mark start/end)
        if axes_n == 1:
            figure_axes["CoG"].plot(
                trial.CoG[:, 0], trial.CoG[:, 1], lw=2, alpha=alpha, color=colors["CoG"]
            )
            figure_axes["CoG"].scatter(
                *trial.CoG[int(data.n_secs_before * data.fps), :],
                zorder=100 if axes_n == 1 else 0,
                s=50,
                color=[0.4, 0.4, 0.4],
                label="movement onset",
            )

        if not data.plot_individual_trials:
            break

    # save and close trial figure
    if data.plot_individual_trials:
        trial_axes["CoG"].legend()
        plt.show(trial_f)

# ------------------------------- plot averages ------------------------------- #
for ch in sensors + ["tot_weight", "CoG"]:
    if ch != "CoG":
        avg = np.mean(np.vstack(data.data[ch].values), axis=0)
        std = np.std(np.vstack(data.data[ch].values), axis=0)
        plot_mean_and_error(avg, std, main_axes[ch], color=dark_colors[ch], lw=4)
    else:
        cog = np.mean(np.dstack([v for v in data.data["CoG"].values]), 2)
        time = np.linspace(-data.n_secs_before, data.n_secs_after, len(cog))
        main_axes[ch].scatter(cog[:, 0], cog[:, 1], c=time, cmap="bwr")

plt.show()
