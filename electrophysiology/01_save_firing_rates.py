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


    try:
        print(units.brain_region.unique())
        units = units.loc[units.brain_region.isin(["PRNr", "PRNc"])]
    except:
        return None, None, None, None, None, None

    # if not len(units):
    #     return None, None, None, None, None, None
    # units = [1, 2, 3]


    # get tracking
    logger.info("Getting tracking")
    recname = recording["name"]
    try:
        left_fl =  pd.Series((TrackingBP & f'name="{recname}"' & "bpname='left_fl'").fetch1())
    except Exception as e:
        logger.info(f"Failed to get tracking: {e}")
        return None, None, None, None, None, None
    right_fl = pd.Series((TrackingBP & f'name="{recname}"' & "bpname='right_fl'").fetch1())
    left_hl = pd.Series((TrackingBP & f'name="{recname}"' & "bpname='left_hl'").fetch1())
    right_hl = pd.Series((TrackingBP & f'name="{recname}"' & "bpname='right_hl'").fetch1())
    body = pd.Series((TrackingBP & f'name="{recname}"' & "bpname='body'").fetch1())

        
    logger.info(f"Got {len(units)} units for {recording['name']}")

    return units, left_fl, right_fl, left_hl, right_hl, body




logger.info("Starting")
for rec in get_recording_names("CUN/PPN"):
    print(f"Processing {rec}")
    tracking_save_path = cache / f"{rec}.parquet"
    # if tracking_save_path.exists():
    #     print("     already done")
    #     continue
    
    # skip recs already done
    date = int(rec.split("_")[1])
    mouse = rec.split("_")[2]
    # if mouse in ["AAA1110750", "BAA110516", "BAA110517", "BAA1110279", "BAA1110281"] or date < 0:
    #     continue
    # if date < 210721:
    #     continue

    units, left_fl, right_fl, left_hl, right_hl, body = get_data(rec)
    if units is None:
        print("Skipping no units")
        continue

    # assert len(units.iloc[1].firing_rate) == len(body.x), f"Rec samples: {len(units.iloc[1].firing_rate)}, tracking: {len(body.x)}"
    


    # save tracking data
    tracking = dict(
        x = body.x,
        y = body.y,
        v = body.bp_speed,
        left_fl_v = left_fl.bp_speed,
        right_fl_v = right_fl.bp_speed,
        left_hl_v = left_hl.bp_speed,
        right_hl_v = right_hl.bp_speed,
    )

    # save units data
    for i, unit in units.iterrows():
        # regions = ["PRNr", "PRNc"]
        regions = ["ICe", "VISp1", "VISp2/3"]
        if unit.brain_region not in regions:
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

