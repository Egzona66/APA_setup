import matplotlib.pyplot as plt

from analysis.fixtures import sensors


def plot_sensors_data(sensors_data: dict):
    f, axes = plt.subplots(figsize=(18, 8), nrows=3, sharex=True)

    for ch, vals in sensors_data.items():
        if ch in sensors:
            axes[0].plot(vals, lw=2, label=ch)
        elif ch == "tot_weight":
            axes[1].plot(vals, lw=2, label=ch)
        else:
            axes[2].plot(vals, lw=2, label=ch)

    for ax in axes:
        ax.axhline(0, lw=5, ls="--", zorder=100, color="k")
        ax.legend()
    plt.show()
