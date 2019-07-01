import sys
sys.path.append("./")

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt


from utils.file_io_utils import *


def plot_frame_delays():
    f = "E:\\Egzona\\test_analoginputs.csv"
    data = load_csv_file(f)

    _, ax = plt.subplots()
    # ax.plot(data.elapsed.values)

    ax.plot(np.diff(data.elapsed.values))
    ax.axhline(np.mean(np.diff(data.elapsed.values)), color="r", lw=2)

    ax.set(xlabel="frame number", ylabel="delta t ms")

    plt.show()

if __name__ == "__main__":
    plot_frame_delays()