import sys
sys.path.append("./")

import cv2
import numpy as np
import matplotlib.pyplot as plt

from dm_control import composer
from dm_control.locomotion.walkers import rodent

from modelling.corridor import Forceplate
from modelling.task import RunThroughCorridor

_CONTROL_TIMESTEP = .02
_PHYSICS_TIMESTEP = 0.001



def rodent_run_gaps(random_state=None):
    """Requires a rodent to run down a corridor with gaps."""

    # Build a position-controlled rodent walker.
    walker = rodent.Rat(
        observable_options={'egocentric_camera': dict(enabled=True)})

    # build forceplate arena
    arena = Forceplate()

    # Build a task that rewards the agent for running down the corridor at a
    # specific velocity.
    task = RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=(.2, 0, 0),
        walker_spawn_rotation=0,
        target_velocity=1.0,
        contact_termination=False,
        terminate_at_height=-0.3,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    return composer.Environment(time_limit=30,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)



def grab_frame(env, camera_id=0):
    # Get RGB rendering of env
    rgbArr = env.physics.render(480, 600, camera_id=camera_id)
    # Convert to BGR for use with OpenCV
    return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)

def grab_frames(env):
    """
        Grab all frames from 4 cameras and stack into a single frame
    """
    frames = []
    for i in range(4):
        frames.append(grab_frame(env, camera_id=i))
    
    # stack frames
    frame = np.concatenate(
        (np.concatenate((frames[0], frames[1]), axis=0),
        np.concatenate((frames[2], frames[3]), axis=0),),
        axis=1)
    return frame

def setup_video(video_name, env, fps=30):
    # Setup video writer - mp4 at 30 fps
    frame = grab_frames(env)
    height, width, _ = frame.shape
    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    return video


def main():
    env = rodent_run_gaps(random_state=42)
    action_spec = env.action_spec()

    def policy(time_step):
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
        time_step = env.step(policy(time_step))
        frame = grab_frames(env)
        video.write(frame)
        print(
            f"reward = {time_step.reward}"
            # f"discount = {time_step.discount}"
            # f"observations = {time_step.observation}"
        )
        step = step + 1
    video.release()
    
    # f, axes = plt.subplots(2, 2, figsize=(10, 10))
    # for (n, ax) in enumerate(axes.flat):
    #     ax.imshow(grab_frame(env, camera_id=n))
    #     ax.set(xticks=[], yticks=[])
    # f.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()