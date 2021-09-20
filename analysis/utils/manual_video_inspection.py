import sys
sys.path.append("./")
from forceplate_config import Config
from fcutils.video.utils import manual_video_inspect

from fcutils.video.utils import get_video_params, get_cap_from_file

# ? To manually inspect a video frame by frame:
# 1) Specify the path to the video you want to analyse
# 2) Run this script

class Inspector(Config,):
    def __init__(self, video_to_inspect):
        Config.__init__(self)

        manual_video_inspect(video_to_inspect)

if __name__ == "__main__":
    videofile = "E:\\Egzona\\2021\\160921_RED_F_1R_3\\160921_RED_F_1R_3_cam0.avi"  # * <--- path to the video to analyse 

    nframes, width, height, fps, is_color = get_video_params(get_cap_from_file(videofile))
    print(f'Video has: {nframes} (wxh: {width} x {height}) at {round(fps, 2)}fps')
    inspector = Inspector(videofile)


    