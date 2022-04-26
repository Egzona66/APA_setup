
import cv2
import numpy as np
from rich.progress import track
from loguru import logger


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

def make_video(model, env, video_name="video.mp4", video_length=150):
    # create cv2 named window
    cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)

    try:
        _env = env.envs[-1].env
        while _env.__class__.__name__ != "Environment":
            _env = _env.env
    except:
        _env = env.unwrapped.env
    video = setup_video(video_name, _env)
    logger.info(f"Writing video to {video_name}")

    # Record the video starting at the first step
    obs = env.reset()
    rew = 0.0
    for i in track(range(video_length + 1), description="Recording video", total=video_length + 1):
        frame = grab_frames(_env)

        # add text to frame with reward value
        rew = rew if isinstance(rew, (float, int)) else rew[0]
        frame = cv2.putText(frame, "rew: "+str(round(rew, 3)), (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # show frame
        cv2.imshow(video_name, frame)
        cv2.waitKey(1)

        video.write(frame)

        # execute next action
        # action = env.action_space.sample()
        action = model.predict(obs)
        try:
            obs, rew, _, _ = env.step(action)
        except:
            # logger.debug("Error in step during video creation")
            break

    # Save the video
    video.release()
    env.close()
    logger.info("Done & saved")

    # Close the window
    cv2.destroyAllWindows()
