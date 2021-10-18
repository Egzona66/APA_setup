import numpy as np
import sys

sys.path.append('./')

from analysis.fixtures import sensors_vectors


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
        k: sensors_data[k].reshape(-1, 1) * vec / 100
        for k, vec in sensors_vectors.items()
    }

    # sum the vectors to get the CoM vector
    CoM = np.sum(np.dstack(scaled_vectors.values()), 2)

    return CoM
