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
import matplotlib.patches as patches


from utils.video_utils import Editor as VideoUtils
from forceplate_config import Config


from utils.file_io_utils import *
from utils.analysis_utils import *
from utils.video_utils import Editor as VideoUtils
from utils.matplotlib_config import *
from utils.plotting_utils import *
from utils.constants import *

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
        csv_file, video_files = parse_folder_files(self.analysis_config["data_folder"], self.analysis_config["experiment_name"])
        self.data = load_csv_file(csv_file)
        normalized = normalize_channel_data(self.data, self.arduino_config["sensors"])

        caps = {k: cv2.VideoCapture(f) for k,f in video_files.items()}


        # Get frame size and FPS for the videos
        video_params = {k:(self.get_video_params(cap)) for k,cap in caps.items()}
        fps = video_params["cam0"][3]

        # Get ffmpeg video writer
        ffmpeg_dict = self.analysis_config["outputdict"]
        ffmpegwriter = skvideo.io.FFmpegWriter(os.path.join(self.analysis_config["data_folder"], "{}_{}.mp4".format(self.analysis_config["experiment_name"], self.analysis_config["clip_name"])), 
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

        f = plt.figure()

        # Start looping over frames
        for framen in tqdm(np.arange(start_frame, start_frame+self.analysis_config["clip_n_frames"])):
            framen = int(framen)
            # Create a figure and save it then close it
            ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=1)
            ax1 = plt.subplot2grid((2, 3), (0, 1), colspan=1)
            ax2 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
            sensorsax = plt.subplot2grid((2, 3), (0, 2), colspan=1, facecolor=[.2, .2, .2])
            gax = plt.subplot2grid((2, 3), (1, 2), colspan=1, facecolor=[.2, .2, .2])


            # Plot frames
            ret, frame0 = caps["cam0"].read()
            ret, frame1 = caps["cam1"].read()

            if not ret: raise FileNotFoundError(video_files)

            downsample = 2
            ax0.imshow(frame0[::downsample,::downsample], interpolation="nearest")
            ax1.imshow(frame1[::downsample,::downsample][::-1, ::-1], interpolation="nearest")


            # Plot sensors traces
            data_range = [int(framen), int(framen+plot_datapoints)]
            channel_rectangles_coords = {"hl":(-1, -1), "fr":(0, 0), "fl":(-1, 0), "hr":(0, -1)}
            allc = {}
            for ch, color in self.analysis_config["plot_colors"].items():
                channel_data = normalized[ch][data_range[0]-50:data_range[1]-50]

                x = np.arange(0, len(channel_data))
                # ax2.fill_between(x, 0, channel_data, color=color, label=ch, alpha=.3)
                ax2.plot(x, channel_data, color=color, label=ch, alpha=.8, lw=5)

                # Plot sensors states as colored rectangles
                channel_data = normalized[ch][data_range[0]:data_range[1]]
                allc[ch] = channel_data
                if channel_data[0] < 0:
                    a = 0
                else:
                    a =channel_data[0]
                fact = 2
                if a < 0 or a*fact > 1: 
                    raise ValueError("a: ", a , "factor ", fact, "alpha", a*fact)
                rect = patches.Rectangle(channel_rectangles_coords[ch], 1, 1, linewidth=1, edgecolor=color,facecolor=color, alpha=a*fact)
                sensorsax.add_patch(rect)
            
            # Decorate sensors ax
            rect = patches.Rectangle([48, 0], 4, 1, linewidth=1, edgecolor=white, facecolor=white, alpha=.5)
            ax2.add_patch(rect)
            ax2.axvline(50, color=white, ls="--", lw=3, alpha=.3)


            # Plot the centre of gravity for the next n frames
            x_pos = (allc["fr"]+allc["hr"]) - (allc["fl"]+allc["hl"])
            y_pos = (allc["fr"]+allc["fl"]) - (allc["hr"]+allc["hl"])

            gax.plot(x_pos, y_pos, alpha=.1, color="w", lw=1)
            gax.scatter(x_pos, y_pos, alpha=.5, cmap="Reds", c=np.arange(len(y_pos)))
            gax.scatter(x_pos[0], y_pos[0], s=100, color="g")
            gax.axvline(0, color="w", ls=":", lw=.5)
            gax.axhline(0, color="w", ls=":", lw=.5)

            # Set axes properties
            ax1.set(xticks=[], yticks=[])
            ax0.set(xticks=[], yticks=[])
            style_legend(ax2)
            ax2.set(xlabel="frames", ylabel="voltage", facecolor=[.2, .2, .2], ylim=[0, 1], xticks=[0, 25, 50, 75, 100], 
                    xticklabels=[framen-50, framen-25, framen, framen+25, framen+50])

            sensorsax.set(xlim=[-1, 1], ylim=[-1, 1], xticks=[-1, -.5, 0, .5, 1], yticks=[-1, -.5, 0, .5, 1])
            gax.set(xlim=[-1, 1], ylim=[-1, 1], xticks=[-1, -.5, 0, .5, 1], yticks=[-1, -.5, 0, .5, 1])

            # convert figure to numpy array and save to video
            # f.draw()
            f.canvas.draw()
            img = np.array(f.canvas.renderer.buffer_rgba())
            ffmpegwriter.writeFrame(img)
            # plt.close()


        # Close ffmpeg writer
        ffmpegwriter.close()

        

if __name__ == "__main__":
    videoplotter = VideoAnalysis()
    videoplotter.composite_video()
    

    