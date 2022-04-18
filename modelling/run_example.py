import sys
sys.path.append("./")

import numpy as np

from modelling.utils import grab_frames, setup_video
from modelling.environment import build_environment

"""
Run the locomotion task with a random policy
"""


env = build_environment(random_state=42)
action_spec = env.action_spec()

def random_policy(time_step):
    return np.random.uniform(
        low=action_spec.minimum,
        high=action_spec.maximum,
        size=action_spec.shape,
    )

# Step through the environment for one episode with random actions.
time_step = env.reset()
max_steps = 20
step = 0
video_name = "video.mp4"
video = setup_video(video_name, env)

while not time_step.last() and step < max_steps:
    time_step = env.step(random_policy(time_step))
    frame = grab_frames(env)
    video.write(frame)
    print(
        f"reward = {time_step.reward}"
        # f"discount = {time_step.discount}"
        # f"observations = {time_step.observation}"
    )
    step = step + 1
video.release()
