import sys
sys.path.append("./")

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings
import cv2
import skvideo.io
from tqdm import tqdm


from utils.video_utils import Editor as VideoUtils
from forceplate_config import Config


from utils.file_io_utils import *
from utils.analysis_utils import *

class VideoAnalysis(Config, VideoUtils):
    def __init__(self):
        VideoUtils.__init__(self)
        Config.__init__(self)

    def animated_plot(self, fps,  n_timepoints = 500):
        # TODO: improve: http://zulko.github.io/blog/2014/11/29/data-animations-with-python-and-moviepy/
        # TODO this doesnt actually wrok !!!
        # Creates an animate plot and saves it as a video

        # get axes range
        # Create ffmpeg writer
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

        # Create figure
        f, ax = plt.subplots()
        ax.set(xlim=[0, n_timepoints], ylim=[-.1, 1.1], facecolor=[.2, .2, .2])

        # Define animation figure
        def animate(i):
            if i > n_timepoints: x_0 = 0
            else: x_0 = i - n_timepoints
                
            trimmed = self.data.iloc[x_0:i]

            for ch in self.arduino_config["sensors"]:
                p = ax.plot(trimmed[ch].values, color=self.analysis_config['plot_colors'][ch])

        # Start animation
        ani = animation.FuncAnimation(f, animate, frames=1000, repeat=True)


        ani.save(os.path.join(self.analysis_config["data_folder"], 'sensors_animatino.mp4'), writer=writer)


    def composite_video(self, start_time_in_secs=True, sensors_plot_height=200, plot_datapoints= 100):
        # plot_datapoints is the n of points for the sensors traces before the current frame to plot 
        # Creates a video with the view of two cameras and the data from the sensors scrolling underneat
        
        # TODO this is currently creating a figure for each frame and then stitching them together in a video file
        # Imporve this shit

        # ! Which folder is being processed is specified in forceplate_config under analysis_config

        # Load data and open videos
        csv_file, video_files = parse_folder_files(self.analysis_config["data_folder"])
        self.data = load_csv_file(csv_file)
        normalized = normalize_channel_data(self.data, self.arduino_config["sensors"])

        caps = {k: cv2.VideoCapture(f) for k,f in video_files.items()}


        # Get frame size and FPS for the videos
        video_params = {k:(self.get_video_params(cap)) for k,cap in caps.items()}
        fps = video_params["cam0"][3]

        # Create animated plot video
        # self.animated_plot(fps) 
        # ? doesnt work

        # Get output clip size and open cv2 videowriter
        # width = video_params["cam0"][1] + video_params["cam1"][1]
        # height = video_params["cam0"][2] + video_params["cam1"][2] + sensors_plot_height
        # dest_filepath = os.path.join(self.analysis_config["data_folder"], "composite_video.mp4")
        # cvwriter = self.open_cvwriter(dest_filepath, w=width, h=height, framerate=fps, iscolor=True)

        # Get ffmpeg video writer
        ffmpeg_dict = self.analysis_config["outputdict"]
        ffmpegwriter = skvideo.io.FFmpegWriter(os.path.join(self.analysis_config["data_folder"], "composite.mp4"), 
                                                        outputdict=ffmpeg_dict)

        # Get clip start time
        if start_time_in_secs:
            start_s = self.analysis_config["start_clip_time_s"]
            start_frame = np.floor(start_s * fps)
        else:
            start_frame = self.analysis_config["start_clip_time_frame"]

        # Move caps to the corresponding frame
        for cap in caps.values(): self.move_cv2cap_to_frame(cap, start_frame)

        # get dest folder
        frames_folder = os.path.join(self.analysis_config["data_folder"], "frames")
        check_create_folder(frames_folder)

        # Start looping over frames
        for framen in tqdm(np.arange(start_frame, start_frame+self.analysis_config["clip_n_frames"])):
            # Create a figure and save it then close it
            f = plt.figure(figsize=(16, 12), facecolor=[.1, .1, .1])
            ax0 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
            ax1 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
            ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

            # Plot frames
            ret, frame = caps["cam0"].read()
            ax0.imshow(frame)
            ret, frame = caps["cam1"].read()
            ax1.imshow(frame[:, ::-1])

            # Plot sensors traces
            data_range = [int(framen), int(framen+plot_datapoints)]
            for ch, color in self.analysis_config["plot_colors"].items():
                channel_data = normalized[ch][data_range[0]:data_range[1]]

                # x = np.linspace(0, len(channel_data), num=len(channel_data))
                x = np.arange(0, len(channel_data))
                ax2.fill_between(x, 0, channel_data, color=color, label=ch, alpha=.3)

            ax2.legend()
            ax2.set(title="Raw Force Sensor Data", xlabel="frames", ylabel="Volts", facecolor=[.2, .2, .2], ylim=[0, 1])

            ax1.set(xticks=[], yticks=[])
            ax0.set(xticks=[], yticks=[])

            # convert figure to numpy array and save to video
            f.canvas.draw()
            img = np.array(f.canvas.renderer.buffer_rgba())
            ffmpegwriter.writeFrame(img)
            plt.close()

        # Close ffmpeg writer
        ffmpegwriter.close()

        # TODO make Y axis fixed
        # TODO x axis label for plo
        # TODO frames titles
        # TODO write in white



        

if __name__ == "__main__":
    videoplotter = VideoAnalysis()
    videoplotter.composite_video()
    

    