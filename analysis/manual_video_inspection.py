import sys

sys.path.append("./")
#from forceplate_config import Config

from fcutils.video import get_video_params, get_cap_from_file
import cv2  # import opencv
import os


def manual_video_inspect(videofilepath):
    """[loads a video and lets the user select manually which frames to show]
            Arguments:
                    videofilepath {[str]} -- [path to video to be opened]
            key bindings:
                    - d: advance to next frame
                    - a: go back to previous frame
                    - s: select frame
                    - f: save frame
    """

    def get_selected_frame(cap, show_frame):
        cap.set(1, show_frame)
        ret, frame = cap.read()  # read the first frame
        return frame

    # Open text file to save selected frames
    fold, name = os.path.split(videofilepath)

    frames_file = open(os.path.join(fold, name.split(".")[0]) + ".txt", "w+")

    cap = cv2.VideoCapture(videofilepath)
    if not cap.isOpened():
        raise FileNotFoundError("Couldnt load the file")

    print(
        """ Instructions
                    - d: advance to next frame
                    - a: go back to previous frame
                    - s: select frame
                    - f: save frame number
                    - q: quit
    """
    )

    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialise showing the first frame
    show_frame = 0
    frame = get_selected_frame(cap, show_frame)

    while True:
        cv2.imshow("frame", frame)

        k = cv2.waitKey(25)

        if k == ord("d"):
            # Display next frame
            if show_frame < number_of_frames:
                show_frame += 1
        elif k == ord("a"):
            # Display the previous frame
            if show_frame > 1:
                show_frame -= 1
        elif k == ord("s"):
            selected_frame = int(input("Enter frame number: "))
            if selected_frame > number_of_frames or selected_frame < 0:
                print(selected_frame, " is an invalid option")
            show_frame = int(selected_frame)
        elif k == ord("f"):
            print("Saving frame to text")
            frames_file.write("\n" + str(show_frame))
        elif k == ord("q"):
            frames_file.close()
            sys.exit()

        try:
            frame = get_selected_frame(cap, show_frame)
            print("Showing frame {} of {}".format(show_frame, number_of_frames))
        except:
            raise ValueError("Could not display frame ", show_frame)


# ? To manually inspect a video frame by frame:
# 1) Specify the path to the video you want to analyse
# 2) Run this script


#class Inspector(Config,):
 #   def __init__(self, video_to_inspect):
  #      Config.__init__(self)

   #     manual_video_inspect(video_to_inspect)


if __name__ == "__main__":
    videofile = "E:\\Egzona\\Forceplate\\2021\\121021_DTR_GREEN\\121021_DTR_GREEN_M_1L-3_cam0.avi"  # * <--- path to the video to analyse

    nframes, width, height, fps, is_color = get_video_params(
        get_cap_from_file(videofile)
    )
    print(f"Video has: {nframes} (wxh: {width} x {height}) at {round(fps, 2)}fps")
    #inspector = Inspector(videofile)
    manual_video_inspect(videofile)