import sys
sys.path.append("./")
from forceplate_config import Config
from utils.video_utils import Editor as VideoUtils


# ? To manually inspect a video frame by frame:
# 1) Specify the path to the video you want to analyse
# 2) Run this script

class Inspector(Config, VideoUtils):
    def __init__(self, video_to_inspect):
        Config.__init__(self)
        VideoUtils.__init__(self)

        self.manual_video_inspect(video_to_inspect, rescale=4)

if __name__ == "__main__":
    videofile = "D:\\Egzona\\2020\\29012020(training)\\M_2R_3_cam0.avi"  # * <--- path to the video to analyse 
    inspector = Inspector(videofile)



