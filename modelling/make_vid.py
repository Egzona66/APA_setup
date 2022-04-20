
import sys
sys.path.append("./")

from rich.progress import track
from stable_baselines3 import A2C

from modelling.utils import grab_frames, setup_video
from modelling.environment import RLEnvironment


model = A2C.load("logs/rl_model_450000_steps.zip")


# make video
video_length = 100
video_name = "video.mp4"
env = RLEnvironment()
video = setup_video(video_name, env.env)

# Record the video starting at the first step
obs = env.reset()
for i in track(range(video_length + 1), description="Recording video", total=video_length + 1):
    action = model.predict(obs)
    obs, _, _, _ = env.step(action)

    frame = grab_frames(env.env)
    video.write(frame)

# Save the video
video.release()
env.close()
print("Done & saved")
