import sys
sys.path.append("./")

from pypylon import pylon
import skvideo.io
import os
import cv2
import numpy as np
import time

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
        # Open FFMPEG camera writers 
        for i in np.arange(self.camera_config["n_cameras"]):
            file_name = os.path.join(self.camera_config["save_folder"], 
                                    self.camera_config["file_name"]+"_{}".format(i)+self.camera_config["video_format"])
            print("Writing to: {}".format(file_name))
            self.cam_writers[i] = skvideo.io.FFmpegWriter(file_name, outputdict=self.camera_config["outputdict"])

    def setup_cameras(self):
        # set up cameras
        for i, cam in enumerate(self.cameras):
            cam.Attach(self.tlFactory.CreateDevice(self.devices[i]))
            print("Using device ", cam.GetDeviceInfo().GetModelName())
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

            print("Exposure time: ", cam.ExposureTime.GetValue())

            # Start grabbing + GRABBING OPTIONS
            cam.Open()
            cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            # ! if you want to extract timestamps for the frames: https://github.com/basler/pypylon/blob/master/samples/grabchunkimage.py

    
            # self.display_frames[i] = cv2.namedWindow("Frame_{}".format(i), cv2.WINDOW_NORMAL)
            # self.grabs[i] = cam.RetrieveResult(self.camera_config["timeout"]*100)

    def stream_videos(self, max_frames=100, debug=False, display=True, fix_fps=False):
            if debug:
                delta_t = [[] for i in range(self.camera_config["n_cameras"])]
                prev_t = [time.time() for i in range(self.camera_config["n_cameras"])]

            image_windows = [pylon.PylonImageWindow() for i in self.cameras]
            for i, window in enumerate(image_windows): window.Create(i)
            
            #self.grab.GrabSucceeded is false when a camera doesnt get a frame
            while True:
                try:

                    if self.frame_count % 100 == 0: print("Frames: ", self.frame_count)

                    # Loop over each camera and get frames
                    # grab = self.cameras.RetrieveResult(self.camera_config["timeout"])  # ? it doesnt work
                    for i, (writer, cam) in enumerate(zip(self.cam_writers.values(), self.cameras)): 

                        grab = cam.RetrieveResult(self.camera_config["timeout"])

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

                    # Update frame count and terminate
                    self.frame_count += 1

                    if max_frames is not None:
                            if self.frame_count >= max_frames: break

                except pylon.TimeoutException as e:
                    print(e)
                    # if self.camera_config["n_cameras"] > 1:
                    #     [writer.close() for writer in self.cam_writers]
                    # else:
                    #     # TODO this doesnt work
                    #     self.cam_writers[0].close()
                    # break

            # Close camera
            for cam in self.cameras: cam.Close()

            if debug:
                return delta_t


if __name__ == "__main__":
    cam = Camera()


