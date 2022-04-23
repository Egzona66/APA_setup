
import sys
sys.path.append("./")


from stable_baselines3 import A2C, DDPG, TD3

from modelling.utils import make_video
from modelling.train import make_env

_model = TD3

model = _model.load("logs/TD3_large_net_relu/rep_1/TD3_large_net_relu_12500_steps.zip")
env = make_env(1)()
make_video(model, env, "video.mp4", video_length=150)

