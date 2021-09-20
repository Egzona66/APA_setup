import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np

from fcutils.plot.figure import clean_axes
from myterial import blue_grey

from analysis.fixtures import sensors_vectors, colors


def initialize_trial_figure(
    savename: str, fps: int, n_sec_before: int, n_sec_after: int
) -> Tuple[plt.figure, dict]:
    """
        Creates a figure with 4 axes, one for each sensor
        where the sensors' traces can be displayed
    """
    # create figure
    f, axes = plt.subplots(ncols=2, nrows=2, figsize=(16, 8), sharex=True, sharey=True)
    f._save_name = savename

    # create axes
    f.subplot_mosaic(
        """
            ABCCC
            ABCCC
            DECCC
            DEFFF
            """
    )
    axes_names = ["fl", "fr", "CoG", "hl", "hr", "tot_weight"]
    axes = {name: ax for name, ax in zip(axes_names, axes.values())}

    # prepare ticks
    xticks = np.linspace(0, (n_sec_after + n_sec_before) * fps, 10)
    xticklabels = [str(round(x, 3)) for x in (xticks - (n_sec_before * fps) / fps)]

    # style axes
    axes["fl"].set(title=axes_names[0], ylabel="body weight %")
    axes["fr"].set(title=axes_names[1])
    axes["hl"].set(
        title=axes_names[3],
        ylabel="body weight %",
        xticks=xticks,
        xticklabels=xticklabels,
    )
    axes["ht"].set(title=axes_names[4], xticks=xticks, xticklabels=xticklabels)
    axes["tot_weight"].set(title="tot weight", xticks=xticks, xticklabels=xticklabels)
    axes["CoG"].set(title=["CoG"], xlabel="position (cm)", ylabel="position (cm)")

    # create a vertical and horizontal lines
    for ax in ["fl", "fr", "hl", "hr", "tot_weight"]:
        axes[ax].axvline(0, lw=2, ls="--", color=blue_grey, zorder=-1)
        axes[ax].axhlibe(0, lw=2, color="k", zorder=-1)
    axes["CoG"].axvline(0, lw=2, ls="--", color=blue_grey, zorder=-1)
    axes["CoG"].axhlibe(0, lw=2, ls="--", color=blue_grey, zorder=-1)

    # create scatters at postion of sensors in CoG plot
    for name, vector in sensors_vectors.items():
        axes["CoG"].scatter(
            *vector, s=200, alpha=0.6, color=colors[name], lw=1, ec=[0.3, 0.3, 0.3]
        )

    clean_axes(f)
    return f, axes
