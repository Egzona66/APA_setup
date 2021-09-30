import numpy as np
import sys
import os
import matplotlib.pyplot as plt


sys.path.append("./")
sys.path.append(os.path.abspath(os.path.join("..")))

from fcutils.plot.elements import plot_mean_and_error
from fcutils.progress import track
from fcutils.maths.coordinates import cart2pol
from myterial.utils import make_palette
from myterial import black,white,green 

from analysis.process_data import DataProcessing
from analysis.fixtures import sensors, colors, dark_colors
from analysis._plot import initialize_trial_figure, initialize_polar_plot_figure, move_figure

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

# initialize pooled trials figure & polar plot figure
main_f, main_axes = initialize_trial_figure(
    "trials", data.fps, data.n_secs_before, data.n_secs_after,
)
polar_f, polar_ax = initialize_polar_plot_figure('CoG_polar')
polar_ax.set(title='CoG')

move_figure(main_f, 10, 1100)
move_figure(polar_f, 1200, 1100)

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
                trial[sensor], lw=1, color=colors[sensor], alpha=alpha, label=sensor,
            )

        # plot CoG (and mark start/end)
        figure_axes["CoG"].plot(
            trial.CoG_centered[:, 0], trial.CoG_centered[:, 1], lw=2, alpha=0.2, color=colors["CoG"], zorder=-1,
        )

        if not data.plot_individual_trials:
            break

    # save and close trial figure
    if data.plot_individual_trials:
        trial_axes["CoG"].legend()
        plt.show(trial_f)
    plt.close(trial_f)

# ------------------------------- plot averages ------------------------------- #
for ch in sensors + ["tot_weight", "CoG"]:
    if ch != "CoG":
        avg = np.mean(np.vstack(data.data[ch].values), axis=0)
        std = np.std(np.vstack(data.data[ch].values), axis=0)
        plot_mean_and_error(avg, std, main_axes[ch], color=dark_colors[ch], lw=4)

        if ch != 'tot_weight':
            plot_mean_and_error(avg, std, main_axes['all'], color=dark_colors[ch], lw=4)
        plt.legend([sensors])

    else:
        # Plot CoG average in Cartesian Coordinates
        cog = np.mean(np.dstack([v for v in data.data["CoG"].values]), 2)
        cog -= cog[0, :]  # centered at the value of t=0
        time = np.linspace(-data.n_secs_before, data.n_secs_after, len(cog))

        palette1 = make_palette(white, black, int(len(cog)/2),)
        palette2 = make_palette(black, white, int(len(cog)/2),)
        colors = palette1 + palette2
        main_axes[ch].scatter(cog[:, 0], cog[:, 1], c=colors, s=100)

        # Plot CoG average in Polar coordinates
        rho, phi = cart2pol(cog[:, 0], cog[:, 1])
        phi = np.radians(phi - 90)

        polar_ax.scatter(
            phi, rho,  c=colors, s=200
        )


plt.show()
