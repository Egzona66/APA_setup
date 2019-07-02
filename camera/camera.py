import sys
sys.path.append("./")

from pypylon import pylon
import skvideo.io
import os
import cv2
import numpy as np
import time

from utils.file_io_utils import *


class Camera():
    def __init__(self):
        self.frame_count = 0
        self.cam_writers = {}
        self.grabs = {}
        self.display_frames = {}

    def start_cameras(self):
        self.get_cameras()  # get the detected cameras
        self.get_camera_writers()   # set up a video grabber for each
        self.setup_cameras()    # set up camera parameters (triggering... )

    def get_cameras(self):
        # Get detected cameras 
        self.tlFactory = pylon.TlFactory.GetInstance()
        self.devices = self.tlFactory.EnumerateDevices()
        if not self.devices: 
            raise ValueError("Could not find any camera")
        else:
            self.cameras = pylon.InstantCameraArray(self.camera_config["n_cameras"])  


    def get_camera_writers(self):
        # Open FFMPEG camera writers if we are saving to video
        if self.camera_config["save_to_video"]: 
            for i, file_name in enumerate(self.video_files_names):
                print("Writing to: {}".format(file_name))
                self.cam_writers[i] = skvideo.io.FFmpegWriter(file_name, outputdict=self.camera_config["outputdict"])
        else:
            self.cam_writers = {str(i):None for i in np.arange(self.camera_config["n_cameras"])}

    def setup_cameras(self):
        # set up cameras
        for i, cam in enumerate(self.cameras):
            cam.Attach(self.tlFactory.CreateDevice(self.devices[i]))
            print("Using camera: ", cam.GetDeviceInfo().GetModelName())
            cam.Open()
            cam.RegisterConfiguration(pylon.ConfigurationEventHandler(), 
                                        pylon.RegistrationMode_ReplaceAll, 
                                        pylon.Cleanup_Delete)

            # Set up Exposure and frame size
            cam.ExposureTime.FromString(self.camera_config["acquisition"]["exposure"])
            cam.Width.FromString(self.camera_config["acquisition"]["frame_width"])
            cam.Height.FromString(self.camera_config["acquisition"]["frame_height"])
            cam.Height.FromString(self.camera_config["acquisition"]["frame_height"])
            cam.Gain.FromString(self.camera_config["acquisition"]["gain"])
            cam.OffsetY.FromString(self.camera_config["acquisition"]["frame_offset_y"])

            # ? Trigger mode set up
            if self.camera_config["trigger_mode"]:
                # Triggering
                cam.TriggerSelector.FromString('FrameStart')
                cam.TriggerMode.FromString('On')
                cam.LineSelector.FromString('Line4')
                cam.LineMode.FromString('Input')
                cam.TriggerSource.FromString('Line4')
                cam.TriggerActivation.FromString('RisingEdge')

                # ! Settings to make sure framerate is correct
                # https://github.com/basler/pypylon/blob/master/samples/grab.py
                cam.OutputQueueSize = 1
                cam.MaxNumBuffer = 3 # Default is 10
            else:
                cam.TriggerMode.FromString("Off")

            # Start grabbing + GRABBING OPTIONS
            cam.Open()
            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            # ! if you want to extract timestamps for the frames: https://github.com/basler/pypylon/blob/master/samples/grabchunkimage.py

    def stream_videos(self, max_frames=None, debug=False):
            display = self.camera_config["live_display"]

            if debug:
                delta_t = [[] for i in range(self.camera_config["n_cameras"])]
                prev_t = [time.time() for i in range(self.camera_config["n_cameras"])]

            if display:
                image_windows = [pylon.PylonImageWindow() for i in self.cameras]
                self.pylon_windows = image_windows
                for i, window in enumerate(image_windows): window.Create(i)
            
            #self.grab.GrabSucceeded is false when a camera doesnt get a frame
            while True:
                try:
                    if self.frame_count % 100 == 0: 
                        # Print the FPS in the last 100 frames
                        if self.frame_count == 0: start = time.time()
                        else:
                            now = time.time()
                            elapsed = now - start
                            start = now

                            # Given that we did 100 frames in elapsedtime, what was the framerate
                            time_per_frame = (elapsed / 100) * 1000
                            fps = round(1000  / time_per_frame, 2) 
                            
                            print("Tot frames: {}, current fps: {}, desired fps {}.".format(
                                        self.frame_count, fps, self.acquisition_framerate))

                    # Loop over each camera and get frames
                    # grab = self.cameras.RetrieveResult(self.camera_config["timeout"])  # ? it doesnt work
                    for i, (writer, cam) in enumerate(zip(self.cam_writers.values(), self.cameras)): 
                        
                        try:
                            grab = cam.RetrieveResult(self.camera_config["timeout"])
                        except:
                            raise ValueError

                        if not grab.GrabSucceeded():
                            break
                        else:
                            # ! writer is disabled
                            if self.camera_config["save_to_video"]:
                                writer.writeFrame(grab.Array)
                            pass

                        if display and self.frame_count % 1 == 0:
                            image_windows[i].SetImage(grab)
                            image_windows[i].Show()

                        if debug:
                            now = time.time()
                            deltat = now-prev_t[i]
                            delta_t[i].append(deltat*1000)
                            prev_t[i] = now

                    # Read the state of the arduino pins and save to file
                    self.read_arduino_write_to_file(grab.TimeStamp)

                    # Update frame count and terminate
                    self.frame_count += 1

                    if max_frames is not None:
                            if self.frame_count >= max_frames: break

                except pylon.TimeoutException as e:
                    print(e)
                    sys.exit()

            # Close camera
            for cam in self.cameras: cam.Close()

            if debug:
                return delta_t

    def close_pylon_windows(self):
        if self.camera_config["live_display"]:
            for window in self.pylon_windows:
                window.Close()

    def close_ffmpeg_writers(self):
        if self.camera_config["save_to_video"]: 
            for writer in self.cam_writers.values():
                writer.close()

    # def close


if __name__ == "__main__":
    cam = Camera()



