import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np

from fcutils.plot.figure import clean_axes
from myterial import blue_grey


def initialize_trial_figure(
    savename: str, fps: int, n_sec_before: int, n_sec_after: int
) -> Tuple[plt.figure, dict]:
    """
        Creates a figure with 4 axes, one for each sensor
        where the sensors' traces can be displayed
    """
    # create figure
    f = plt.figure(figsize=(16, 8))
    f._save_name = savename

    # create axes
    axes = f.subplot_mosaic(
        """
            AABBCCC
            AABBCCC
            DDEECCC
            DDEEFFF
            """
    )
    axes_names = [
        ("A", "fl"),
        ("B", "fr"),
        ("C", "CoG"),
        ("D", "hl"),
        ("E", "hr"),
        ("F", "tot_weight"),
    ]
    axes = {name: axes[letter] for letter, name in axes_names}

    # prepare ticks
    xticks = np.linspace(0, (n_sec_after + n_sec_before) * fps, 5).astype(np.int32)
    xticklabels = [str(round(x, 3)) for x in (xticks / fps) - n_sec_before]

    # style axes
    axes["fl"].set(
        title="fl",
        ylabel="body weight %",
        xticks=xticks,
        xticklabels=xticklabels,
        xlabel="time (s)",
        ylim=[-5, 100],
    )
    axes["fr"].set(
        title="fr",
        xticks=xticks,
        xticklabels=xticklabels,
        xlabel="time (s)",
        ylim=[-5, 100],
    )
    axes["hl"].set(
        title="hl",
        ylabel="body weight %",
        xticks=xticks,
        xticklabels=xticklabels,
        xlabel="time (s)",
        ylim=[-5, 100],
    )
    axes["hr"].set(
        title="hr",
        xticks=xticks,
        xticklabels=xticklabels,
        xlabel="time (s)",
        ylim=[-5, 100],
    )
    axes["tot_weight"].set(
        title="tot weight",
        xticks=xticks,
        xticklabels=xticklabels,
        xlabel="time (s)",
        ylabel="tot weight %",
    )
    axes["CoG"].set(title="CoG", xlabel="position (cm)", ylabel="position (cm)")
    axes["CoG"].grid(True)

    # create a vertical and horizontal lines
    for ax in ["fl", "fr", "hl", "hr", "tot_weight"]:
        axes[ax].axvline(
            int(n_sec_before * fps), lw=2, ls="--", color=blue_grey, zorder=-1
        )
        axes[ax].axhline(0, lw=2, color="k", zorder=-1)
    # axes["CoG"].axvline(0, lw=2, ls="--", color=blue_grey, zorder=-1)
    # axes["CoG"].axhline(0, lw=2, ls="--", color=blue_grey, zorder=-1)
    axes["tot_weight"].axhline(0, lw=2, ls=":", color=[0.2, 0.2, 0.2], zorder=-1)

    # create scatters at postion of sensors in CoG plot
    # for name, vector in sensors_vectors.items():
    #     axes["CoG"].scatter(
    #         *vector, s=200, color=colors[name], lw=1, ec=[0.3, 0.3, 0.3]
    # )

    f.tight_layout()
    clean_axes(f)
    return f, axes
