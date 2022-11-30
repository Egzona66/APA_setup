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
    TrackingBP,
    Probe,
    Unit,
    Recording,
    CCM
)
from data.dbase._tracking import load_dlc_tracking, process_body_part


cache = Path(r"J:\APA")



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
    # get units
    recording = (Recording & f"name='{recording}'").fetch1()
    cf = recording["recording_probe_configuration"]
    units = Unit.get_session_units(
        recording["name"],
        cf,
        spikes=True,
        firing_rate=True,
        frate_window=250,
    )

    print(units.head())
    try:
        units = units.loc[units.brain_region.isin(["PRNr", "PRNc"])]
    except:
        return None, None, None, None, None, None
    if not len(units):
        return None, None, None, None, None, None


    # get tracking
    try:
        left_fl =  pd.Series((TrackingBP & f'name="{recording}"' & "bpname='left_fl'").fetch1())
    except:
        return None, None, None, None, None, None
    right_fl = pd.Series((TrackingBP & f'name="{recording}"' & "bpname='right_fl'").fetch1())
    left_hl = pd.Series((TrackingBP & f'name="{recording}"' & "bpname='left_hl'").fetch1())
    right_hl = pd.Series((TrackingBP & f'name="{recording}"' & "bpname='right_hl'").fetch1())
    body = pd.Series((TrackingBP & f'name="{recording}"' & "bpname='body'").fetch1())

        
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
    tracking_save_path = cache / f"{rec}.parquet"
    if tracking_save_path.exists():
        continue
    
    # skip recs already done
    date = int(rec.split("_")[1])
    mouse = rec.split("_")[2]
    if mouse in ["BAA110516", "BAA110517", "AAA1110750", "BAA1110279"] or date < 210916:
        continue

    units, left_fl, right_fl, left_hl, right_hl, body = get_data(rec)
    if units is None:
        continue


    # save tracking data
    tracking = dict(
        x = upsample_frames_to_ms(body.x),
        y = upsample_frames_to_ms(body.y),
        v = upsample_frames_to_ms(body.bp_speed),
        left_fl_v = upsample_frames_to_ms(left_fl.bp_speed),
        right_fl_v = upsample_frames_to_ms(right_fl.bp_speed),
        left_hl_v = upsample_frames_to_ms(left_hl.bp_speed),
        right_hl_v = upsample_frames_to_ms(right_hl.bp_speed),
    )

    # save units data
    for i, unit in units.iterrows():
        if unit.brain_region not in ["PRNr", "PRNc"]:
            continue
        assert len
        name = f"{rec}_{unit.unit_id}_{unit.brain_region}.npy"
        unit_save = cache / name

        # get firing rate
        if not unit_save.exists():
            try:
                np.save(unit_save, unit.firing_rate)
            except:
                logger.warning(f"Could not save {unit_save}")
                continue
        # else:
        #     fr = np.load(unit_save)

    pd.DataFrame(tracking).to_parquet(tracking_save_path) 


