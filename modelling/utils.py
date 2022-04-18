
import cv2
import numpy as np

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