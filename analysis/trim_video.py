
import sys
sys.path.append("./")

from pathlib import Path
from fcutils.video import trim_clip
from analysis.process_data import DataProcessing

n_secs_before = .5
n_secs_after = .5 

save_fld = Path("/Volumes/EGZONA/Egzona/Forceplate/DLC/clips")

data = DataProcessing.reload().data
print(data)

for i, trial in data.iterrows():
    if not trial.video.exists():
        print(f"Could not find video for trial: {trial.video}")
        continue

    # get start frame in original FPS
    frame = int(
        trial.movement_onset_frame * trial.original_fps / 600
    )

    n_frames_before = int(n_secs_before * 600)
    n_frames_after = int(n_secs_after * 600)

    save_path = save_fld / f"{trial['name']}_{trial.movement_onset_frame}.mp4"
    trim_clip(
        str(trial.video),
        save_path,
        start_frame=frame - n_frames_before,
        end_frame=frame + n_frames_after
    )
    print(f"Saved clip for trial: {trial.video}")


