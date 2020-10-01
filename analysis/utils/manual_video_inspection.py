import sys
sys.path.append("./")
from forceplate_config import Config
from fcutils.video.utils import manual_video_inspect

# ? To manually inspect a video frame by frame:
# 1) Specify the path to the video you want to analyse
# 2) Run this script

class Inspector(Config,):
    def __init__(self, video_to_inspect):
        Config.__init__(self)

        manual_video_inspect(video_to_inspect)

if __name__ == "__main__":
    videofile = "D:\\Egzona\\Forceplate\\2020\\27022020\\27022020_M_2R_2_cam0.avi"  # * <--- path to the video to analyse 
    inspector = Inspector(videofile)

