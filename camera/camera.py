import sys
sys.path.append("./")

from pypylon import pylon
import skvideo.io
import os
import cv2
import numpy as np

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
            self.cameras = pylon.InstantCameraArray(self.camera_config["n_cameras"]) # ! change this to use multiple cameras

    def get_camera_writers(self):
        # Open FFMPEG camera writers 
        for i in np.arange(self.camera_config["n_cameras"]):
            file_name = os.path.join(self.camera_config["save_folder"], 
                                    self.camera_config["file_name"]+"_{}".format(i)+self.camera_config["video_format"])
            print("Writing to: {}".format(file_name))
            self.cam_writers[i] = skvideo.io.FFmpegWriter(file_name, outputdict=self.camera_config["outputdict"])

    def setup_cameras(self):
        for i, cam in enumerate(self.cameras):

            cam.Attach(self.tlFactory.CreateDevice(self.devices[i]))
            print("Using device ", cam.GetDeviceInfo().GetModelName())
            cam.Open()
            cam.RegisterConfiguration(pylon.ConfigurationEventHandler(), 
                                        pylon.RegistrationMode_ReplaceAll, 
                                        pylon.Cleanup_Delete)

            # ? SL set upts
            # cam.TriggerSelector.FromString('FrameStart')
            # cam.TriggerMode.FromString('On')
            # cam.LineSelector.FromString('Line3')
            # cam.LineMode.FromString('Input')
            # cam.TriggerSource.FromString('Line3')
            # cam.TriggerActivation.FromString('RisingEdge')

            cam.TriggerMode.FromString("Off")

            # Start grabbing
            cam.StartGrabbing()
    
            self.display_frames[i] = cv2.namedWindow("Frame_{}".format(i), cv2.WINDOW_NORMAL)
            self.grabs[i] = cam.RetrieveResult(self.camera_config["timeout"]*100)

    def stream_videos(self):
        #self.grab.GrabSucceeded is false when a camera doesnt get a frame
        while True:
            try:
                grabbers = list(self.grabs.values()).copy()
                for i, (writer, grab, cam) in enumerate(zip(self.cam_writers.values(), grabbers, self.cameras)): # Loop over each camera and get frames
                    writer.writeFrame(grab.Array)

                    if self.frame_count % 10 == 0:
                        print("Frames: ", self.frame_count)
                        cv2.imshow("Frame_{}".format(i), grab.Array)
                        cv2.waitKey(1)
                    self.frame_count += 1

                    grab = cam.RetrieveResult(self.camera_config["timeout"])
                    if not grab.GrabSucceeded():
                        break
                    else:
                        self.grabs[i] = grab

            except pylon.TimeoutException as e:
                print(e)
                [writer.close() for writer in self.cam_writers]
                break




if __name__ == "__main__":
    cam = Camera()


