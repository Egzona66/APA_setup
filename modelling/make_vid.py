
import sys
sys.path.append("./")

import cv2
from rich.progress import track
from stable_baselines3 import A2C

from modelling.utils import grab_frames, setup_video
from modelling.environment import RLEnvironment
from modelling.train import make_env


model = A2C.load("logs/long/rep_1/long_180000_steps.zip")


# make video
video_length = 150
video_name = "video.mp4"
env = make_env(1)()

# create cv2 named window
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

_env = env.unwrapped.env
video = setup_video(video_name, _env)

# Record the video starting at the first step
obs = env.reset()
rew = 0.0
for i in track(range(video_length + 1), description="Recording video", total=video_length + 1):
    frame = grab_frames(_env)

    # add text to frame with reward value
    frame = cv2.putText(frame, "rew: "+str(round(rew, 3)), (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # show frame
    cv2.imshow("frame", frame)
    cv2.waitKey(1)

    video.write(frame)

    # execute next action
    action = env.action_space.sample()
    try:
        obs, rew, _, _ = env.step(action)
    except:
        print("Error in step")
        break

# Save the video
video.release()
env.close()
print("Done & saved")
