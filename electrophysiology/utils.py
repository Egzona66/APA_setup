import pandas as pd
import numpy as np
from pathlib import Path

from fcutils.maths import derivative
from fcutils.maths.signals import get_onset_offset



# --------------------------------- load data -------------------------------- #
def load_rec_data(cache, rec:str)->pd.DataFrame:
    """
        Load the units for a recording
    """
    units_files = list(cache.glob(f"{rec}*.npy"))
    unit_names = [f.stem.split("_")[-1] for f in units_files]
    unit_data = [np.load(f) for f in units_files]

    units = pd.DataFrame({k:v for k,v in zip(unit_names, unit_data)})
    tracking = pd.read_parquet(cache / f"{rec}.parquet")
    return tracking, units




# --------------------------------- movement --------------------------------- #
def find_starting_paw(tracking:pd.DataFrame, start:int) -> str:
    paw_values = dict(
        left_fl_v = tracking.left_fl_v[start],
        left_hl_v = tracking.left_hl_v[start],
        right_fl_v = tracking.right_fl_v[start],
        right_hl_v = tracking.right_hl_v[start]
    )
    starter = list(paw_values.keys())[np.argmax(list(paw_values.values()))]
    return starter


def get_locomotion_onset_times(
            tracking: pd.DataFrame, 
            paw:str="right_fl", 
            cutoff=2000,
            movement_th = 9,
    ) -> list:
    """
        Finds the precise timing of locomotion onsets from tracking data, looking
        specifically for the movement of the selected paw.
    """

    # get when each of the paw is movign
    def is_it_moving(v):
        is_moving = np.where(tracking.v > movement_th)[0]
        moving = np.zeros_like(tracking.v)
        moving[is_moving] = 1

    left_fl = is_it_moving(tracking.left_fl_v)
    right_fl = is_it_moving(tracking.right_fl_v)
    left_hl = is_it_moving(tracking.left_hl_v)
    right_hl = is_it_moving(tracking.right_hl_v)
    
    # check how many paws are moving
    moving = np.sum([left_fl, right_fl, left_hl, right_hl], axis=0)
    moving[moving < 2] = 0
    moving[moving >= 2] = 1

    # get start/stop of movement
    move_start, _ = get_onset_offset(moving, .5)
    v = tracking.v.values
    d = derivative(v)


    selected = 0
    timepoints = []
    for start in move_start:
        if start < cutoff:
            continue
        # exclude baddies
        if np.mean(v[start-cutoff:start]) > 5:
            continue
        
        if np.mean(v[start:start+1000]) < 5:
            continue

        if np.max(v[start-cutoff:start]) >= 25:
            continue

        if np.min(v[start+200:start+1000]) < 5:
            continue

        if len(timepoints) and start - timepoints[-1] < cutoff:
            continue

        # find precise start time based on derivative
        _d = d[start-500:start]
        try:
            shift = np.where(_d < 0)[0][-1]
        except IndexError:
            shift = 500
        precise_start = start - (500 - shift)

        # make sure it's the right paw
        starter = find_starting_paw(tracking, precise_start)
        if paw not in starter:
            continue
        selected += 1

        timepoints.append(precise_start)

    return timepoints