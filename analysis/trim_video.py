from fcutils.video import trim_clip

import sys
sys.path.append("./")
from analysis.process_data import DataProcessing

data = DataProcessing.reload()
print(data[["name", "movement_onset_frame", "video"]])

# video = "path/to/video.mp4"
# savepath  = "path/to/video_to_save.mp4"

# FRAME = 2131123  # trial start frame
# nframes_pre = 250
# nframes_post = 250

# trim_clip(video, savepath, start=FRAME - nframes_pre, end=FRAME + nframes_post)