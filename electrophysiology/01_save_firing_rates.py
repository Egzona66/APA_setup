# imports
from concurrent.futures import process
import sys
from fcutils.maths.signals import rolling_mean
from fcutils.maths import derivative
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import interpolate
from loguru import logger
from scipy.ndimage.filters import gaussian_filter1d


sys.path.append("./")
sys.path.append(r"C:\Users\Federico\Documents\GitHub\pysical_locomotion")

from data.dbase.db_tables import (
    Probe,
    Unit,
    Recording,
    CCM
)
from data.dbase._tracking import load_dlc_tracking, process_body_part


cache = Path(r"J:\APA")


def process_tracking_data(recording: str):
    recfile = list(Path(r"K:\tracking").glob(f"{recording}*.h5"))[0]

    if not Path(recfile).exists():
        logger.warning(f"File {recfile} does not exist")
        return None

    body_parts_tracking: dict = load_dlc_tracking(recfile)
    body_parts_tracking = {
        k: v for k, v in body_parts_tracking.items() if k in ("body", "tail_base", "left_fl", "right_fl", "left_hl", "right_hl")
    }

    M = (CCM & f"name='{recording}'").fetch1("correction_matrix")
    body_parts_tracking = {
        k: process_body_part(
            k, bp, M, likelihood_th=.95, cm_per_px=60/830
        )
        for k, bp in body_parts_tracking.items()
    }
    for k in body_parts_tracking.keys():
        body_parts_tracking[k]['speed'] = body_parts_tracking[k]['bp_speed']
        del body_parts_tracking[k]['bp_speed']
    return body_parts_tracking


def get_recording_names(region="MOs"):
    """
        Get the names of the recordings (M2)
    """
    if region == "MOs":
        return (Recording * Probe & "target='MOs'").fetch("name")
    else:
        return (Recording * Probe - "target='MOs'").fetch("name")


def get_data(recording: str):
    """
        Get all relevant data for a recording.
        Gets the ephys data and tracking data for all limbs
    """
    body_parts_tracking = process_tracking_data(recording)
    if body_parts_tracking is None:
        return None, None, None, None, None, None

    left_fl = pd.Series(body_parts_tracking["left_fl"])
    right_fl = pd.Series(body_parts_tracking["right_fl"])
    left_hl = pd.Series(body_parts_tracking["left_hl"])
    right_hl = pd.Series(body_parts_tracking["right_hl"])
    body = pd.Series(body_parts_tracking["body"])


    # get units
    recording = (Recording & f"name='{recording}'").fetch1()
    cf = recording["recording_probe_configuration"]
    units = Unit.get_session_units(
        recording["name"],
        cf,
        spikes=True,
        firing_rate=False,
        frate_window=100,
    )

    if len(units):
        units = units.sort_values("brain_region", inplace=False).reset_index()
        # units = units.loc[units.brain_region.isin(("PRNr", "PRNc"))]
    else:
        return None, None, None, None, None, None
        
    logger.info(f"Got {len(units)} units for {recording['name']}")

    return units, left_fl, right_fl, left_hl, right_hl, body

def calc_firing_rate(spikes_train: np.ndarray, dt: int = 10):
    """
        Computes the firing rate given a spikes train (wether there is a spike or not at each ms).
        Using a gaussian kernel with standard deviation = dt/2 [dt is in ms]
    """
    return gaussian_filter1d(spikes_train, dt)  * 1000


# Process data
def upsample_frames_to_ms(var):
    """
        Interpolates the values of a variable expressed in frams (60 fps)
        to values expressed in milliseconds.
    """
    t_60fps = np.arange(len(var)) / 60
    f = interpolate.interp1d(t_60fps, var)

    t_1000fps = np.arange(0, t_60fps[-1], step=1/1000)
    interpolated_variable_values = f(t_1000fps)
    return interpolated_variable_values





for rec in get_recording_names("CUN/PPN"):
    print(f"Processing {rec}")
    units, left_fl, right_fl, left_hl, right_hl, body = get_data(rec)
    if units is None:
        continue


    # save tracking data
    tracking = dict(
        x = upsample_frames_to_ms(body.x),
        y = upsample_frames_to_ms(body.y),
        v = upsample_frames_to_ms(body.speed),
        left_fl_v = upsample_frames_to_ms(left_fl.speed),
        right_fl_v = upsample_frames_to_ms(right_fl.speed),
        left_hl_v = upsample_frames_to_ms(left_hl.speed),
        right_hl_v = upsample_frames_to_ms(right_hl.speed),
    )
    pd.DataFrame(tracking).to_parquet(cache / f"{rec}.parquet") 

    # save units data
    for i, unit in units.iterrows():
        if unit.brain_region not in ["PRNr", "PRNc"]:
            continue
        name = f"{rec}_{unit.unit_id}_{unit.brain_region}.npy"
        unit_save = cache / name

        # get firing rate
        if not unit_save.exists():
            time = np.zeros(len(tracking["v"]))  # time in milliseconds
            spikes_times = np.int64(np.round(unit.spikes_ms))
            spikes_times = spikes_times[spikes_times < len(time)]
            time[spikes_times] = 1

            fr = calc_firing_rate(time, dt=50)  
            try:
                np.save(unit_save, fr)
            except:
                logger.warning(f"Could not save {unit_save}")
                continue
        else:
            fr = np.load(unit_save)

