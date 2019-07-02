import sys
sys.path.append("./")

import os
import numpy as np


def parse_folder_files(folder):
    video_files = {}
    for f in os.listdir(folder):
        if "csv" in f:
            csv_file = os.path.join(folder, f)
        elif "cam0" in f:
            video_files["cam0"] = os.path.join(folder, f)
        elif "cam1" in f:
            video_files["cam1"] = os.path.join(folder, f)

    return csv_file, video_files

def normalize_channel_data(data, sensors):
    return  {ch: data[ch].values - np.nanmedian(data[ch].values) for ch in sensors}