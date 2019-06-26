import sys
sys.path.append("./")

import os
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt

from camera import Camera


class CameraTest(Camera):
    # Camera Setup Options
    camera_config = {
        "save_to_video": False,
        "video_format": ".avi",
        "save_folder": "F:\\Egzona", # where you want to save the file
        "file_name": "python_camera",
        "outputdict":{
            '-vcodec': 'libx264',
            '-crf': '0',
            '-preset': 'slow',
            '-pix_fmt': 'yuvj444p',
            '-framerate': '30'  # ! <- framerate of output video. FPS of acquisition is hardware triggered
        },

        "n_cameras": 2,
        "timeout": 100,   # frame acquisition timeout

        "trigger_mode":True,  # hardware triggering
        "acquisition": {    # ? acquisition params  
            # ! if you want to be able to specify these params differently for the different cameraas
            # ask Federico
            "exposure": "5000",
            "frame_width": "960",  # must be a multiple of 32
            "frame_height": "960", # must be a multiple of 32
            "gain": "10"
        }

    }

    # ? camera testing options
    n_frames_test = 1500

    def __init__(self):
        Camera.__init__(self)
        self.start_cameras()

    def test_camera_trigger(self):
        self.stream_videos(max_frames = 10, display=True, debug=False)


    def test_cameras(self):
        start = time.time()

        delta_t = self.stream_videos(max_frames = self.n_frames_test, display=True, debug=True)
        end = time.time()
        approx_fps = round(self.frame_count / (end-start), 2)
        approx_deltat = 1000/approx_fps


        # Print results
        print("""
            Number of frames to acquire: {}
            Number of frames acquired:   {}
            Time elapsed:                {}
            Approx FPS:                  {}
            Delta T mean std:            ({}, {})
        """.format(self.n_frames_test, self.frame_count, end-start, approx_fps, np.mean(delta_t[0]), np.std(delta_t[0])))

        # Plot stuff
        f, axarr = plt.subplots(ncols=2, sharex=True)
        axarr[0].plot(delta_t[0], color="m", label="1")
        axarr[0].axhline(approx_deltat, color="r", lw=2, label="dT", alpha=1)

        axarr[0].set(title="frames delta t", xlabel="frames", ylabel="delta T", facecolor=[.2,.2,.2])

        axarr[1].plot(np.cumsum(delta_t[0]), color="m", label="1")
        axarr[1].set(title="frames delta t", xlabel="time", ylabel="delta T", facecolor=[.2,.2,.2])

        if self.camera_config["n_cameras"] > 1:
            axarr[0].plot(delta_t[1], color="g", label="2")
            axarr[1].plot(np.cumsum(delta_t[1]), color="g", label="2")

    def get_videos_size(self):
        # Get the frame size, number of frames and fps of the saved test videos
        videos = [os.path.join(self.camera_config["save_folder"], v) for v in os.listdir(self.camera_config["save_folder"])]

        for video in videos:
            cap = cv2.VideoCapture(video)
            if not cap.isOpened(): raise FileNotFoundError("Could not open video: ", video)

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print("""
    Video:       {}
    fps:         {}
    frame size: ({}, {})
    n_frames:    {}
            
            """.format(os.path.split(video)[1], fps, w,h, n_frames))



if __name__ == "__main__":
    test = CameraTest()
    test.test_cameras()
    # test.get_videos_size()

    # test.test_camera_trigger()

    plt.show()