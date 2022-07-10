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
from myterial import black,white,green, blue_light, salmon, blue_dark

from analysis.process_data import DataProcessing
from analysis.fixtures import sensors, colors, dark_colors
from analysis._plot import initialize_trial_figure, initialize_polar_plot_figure, move_figure

"""
    Plots individual trials and population averages.
    Current plots:  
        - CoM traces
        - indidivual sensors traces
"""
USE_COG = "CoG_centered"  # or CoG_centered

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
            trial[USE_COG][:, 0], trial[USE_COG][:, 1], lw=2, alpha=0.2, color=colors["CoG"], zorder=-1,
        )
        figure_axes['X'].plot(trial[USE_COG][:, 0], lw=2, alpha=.2, color=colors["CoG"], zorder=-1)
        figure_axes['Y'].plot(trial[USE_COG][:, 1], lw=2, alpha=.2, color=colors["CoG"], zorder=-1)
        figure_axes['S'].scatter(trial.CoG[0, 0], trial.CoG[0,1], alpha=1, color=colors["CoG"], zorder=-1)

        # also on polar plot
        rho, phi = cart2pol(trial[USE_COG][0, 0], trial[USE_COG][0, 1])
        phi = np.radians(phi - 90)

        polar_ax.scatter(
            phi, rho,  alpha=1, color=colors["CoG"], zorder=-1, s=200, label=None
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
        main_axes['all'].legend([sensors])

    else:
        # Plot CoG average in Cartesian Coordinates
        COGs = np.dstack([v for v in data.data[USE_COG].values])
        cog = np.mean(COGs, 2)
        cog_std = np.std(COGs, 2)

        # cog -= cog[0, :]  # centered at the value of t=0
        time = np.linspace(-data.n_secs_before, data.n_secs_after, len(cog))

        palette1 = make_palette(blue_light, blue_dark, int(len(cog)/2),)
        palette2 = make_palette(blue_dark, white, int(len(cog)/2),)
        colors = palette1 + palette2
        main_axes[ch].plot(cog[:, 0], cog[:, 1], c=[.3, .3, .3], lw=3)
        main_axes[ch].scatter(cog[0, 0], cog[0, 1], c=[.3, .3, .3], ec=[.3, .3, .3], lw=3, s=100, zorder=100)
        main_axes[ch].scatter(cog[-1, 0], cog[-1, 1], c='white', ec=[.3, .3, .3], lw=3, s=100, zorder=100)

        # plot COG XY traces separately
        plot_mean_and_error(cog[:, 0], cog_std[:, 0], main_axes['X'])
        plot_mean_and_error(cog[:, 1], cog_std[:, 1], main_axes['Y'])

        # main_axes['X'].scatter(np.arange(len(cog)), cog[:, 0], c=colors, s=10)
        # main_axes['Y'].scatter(np.arange(len(cog)), cog[:, 1],  c=colors, s=10)

        # Plot CoG average in Polar coordinates
        rho, phi = cart2pol(cog[:, 0], cog[:, 1])
        phi = np.radians(phi - 90)

        polar_ax.scatter(
            phi, rho,  c=colors, s=200
        )


# save figure
strain = "".join(data.params["STRAINS"])
condition = "".join(data.params["CONDITIONS"])
main_f.savefig(f"C:\\Users\\Federico\\Desktop\\forE\\trial_traces_strain_{strain}_condition_{condition}.png", dpi=300)
polar_f.savefig(f"C:\\Users\\Federico\\Desktop\\forE\\trial_traces_strain_polar_plot_{strain}_condition_{condition}.png", dpi=300)

plt.show()
