from pypylon import pylon
import skvideo.io

TIMEOUT_LIMIT = 100

def run_camera_acquisition(save_dir):
    outputdict = {
        '-vcodec': 'libx264',
        '-crf': '0',
        '-preset': 'slow',
        '-pix_fmt': 'yuvj444p',
        '-framerate': '30'
    }

    maxCamerasToUse = 1
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()

    cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

    file_name = "{}\\camera.avi".format(save_dir)
    print("Writing to: {}".format(file_name), cameras[0].GetDeviceInfo().GetModelName())
    cam_writer = skvideo.io.FFmpegWriter(file_name, outputdict=outputdict)

    for i, cam in enumerate(cameras):

        cam.Attach(tlFactory.CreateDevice(devices[i]))
        print("Using device ", cam.GetDeviceInfo().GetModelName())
        cam.Open()
        cam.RegisterConfiguration(pylon.ConfigurationEventHandler(), pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_Delete)
        cam.TriggerSelector.FromString('FrameStart')
        cam.TriggerMode.FromString('On')
        cam.LineSelector.FromString('Line3')
        cam.LineMode.FromString('Input')
        cam.TriggerSource.FromString('Line3')
        cam.TriggerActivation.FromString('RisingEdge')

    cameras.StartGrabbing()

    imageWindow1 = pylon.PylonImageWindow()
    imageWindow1.Create(1)
    GrabResult1 = cameras[0].RetrieveResult(TIMEOUT_LIMIT*100)

    while GrabResult1.GrabSucceeded:
        try:
            cam_writer.writeFrame(GrabResult1.Array)
            imageWindow1.SetImage(GrabResult1)
            imageWindow1.Show()
            GrabResult1 = cameras[0].RetrieveResult(TIMEOUT_LIMIT)

        except pylon.TimeoutException as e:
            print(e)
            cam_writer.close()
            break

if __name__ == '__main__':
    run_camera_acquisition('test')