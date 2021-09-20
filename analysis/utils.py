from pathlib import Path
import os
from typing import Tuple
import numpy as np

from analysis.fixtures import sensors_vectors


def parse_folder_files(folder: Path, exp_name: str) -> Tuple[Path, Path]:
    folder = str(folder)
    video_files = {}
    for f in os.listdir(folder):
        if exp_name not in f:
            continue

        if "csv" in f:
            csv_file = os.path.join(folder, f)
        elif "cam0" in f and "txt" not in f:
            video_files["cam0"] = os.path.join(folder, f)
        # elif "cam1" in f and "txt" not in f:
        # video_files["cam1"] = os.path.join(folder, f)
    try:
        return Path(csv_file), Path(video_files)
    except:
        raise FileNotFoundError(
            f"Could not fine csv or video files for {exp_name} in folder {folder}"
        )


def correct_paw_used(sensors_data, paw_used):
    if "l" in paw_used.lower():
        return {
            "fr": sensors_data["fl"],
            "fl": sensors_data["fr"],
            "hr": sensors_data["hl"],
            "hl": sensors_data["hr"],
        }
    return sensors_data


def compute_cog(sensors_data: dict) -> np.ndarray:
    """
        Computs the position of the center of gravity.
        Each sensor is a 2d vector indicating it's position in space.
        At each frame, each sensor's vector magnitude is scaled by the 
        ammount of weight (in %) recorded by the sensor.
        The sensors vectors are then summed to get a vector indicating the displacement of
        the CoG
    """

    # scale each vector by the % of weight on the sensor on each frame
    # so that each scaled vector is in the 0-1 range
    scaled_vectors = {
        k: sensors_data[k] * vec / 100 for k, vec in sensors_vectors.items()
    }

    # sum the vectors to get the CoM vector
    CoM = np.sum(np.vstack(scaled_vectors.values()), 1)

    return CoM
