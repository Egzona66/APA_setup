
import sys
sys.path.append("./")


import matplotlib.pyplot as plt
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from stable_baselines3 import A2C
from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common import results_plotter

import torch as th
from rich.progress import track

from modelling.utils import grab_frames, setup_video
from modelling.environment import RLEnvironment
from modelling.networks import ProprioceptiveEncoder


def make_env(rank, seed=0, log_dir=None):
    """
    Utility function for multiprocessed env.

    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RLEnvironment()
        
        env.seed(seed + rank)
        log_file = os.path.join(log_dir, str(rank)) if log_dir is not None else None
        env = Monitor(env, log_file)
        return env

    set_random_seed(seed)
    return _init



def make_agent(env, tensorboard_log=None):
    # policy_kwargs = dict(
    #     features_extractor_class=ProprioceptiveEncoder,
    #     features_extractor_kwargs=dict(features_dim=8),
    # )

    policy_kwargs = dict(activation_fn=th.nn.Tanh,
                        net_arch=[dict(pi=[256, 256], vf=[256, 256])])

    model = A2C('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)

    # model = RecurrentPPO(
    #         "MlpLstmPolicy",
    #         env=env,
    #         n_steps=128,
    #         batch_size=14,
    #         gamma=0.5,
    #         n_epochs=5,
    #         ent_coef=0.0,
    #         policy_kwargs=policy_kwargs,
    #         tensorboard_log=tensorboard_log,
    #         verbose=1,
    # )
    return model


if __name__ == '__main__':
    TRAIN = True
    N_CPU = 10
    N_STEPS = 20_000

    # TODO create input network to compress inputs

    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    if TRAIN:
        checkpoint_callback = CheckpointCallback(save_freq=2500, save_path=log_dir,
                                         name_prefix='rl_model')

        # Create the vectorized environment
        if N_CPU > 1:
            env = DummyVecEnv([make_env(i, log_dir=log_dir) for i in range(N_CPU)])  # or SubprocVecEnv/DummyVecEnv
        else:
            env = Monitor(RLEnvironment(), log_dir)
            check_env(env)
        
        model = make_agent(env, tensorboard_log=log_dir)
        model.learn(total_timesteps=N_STEPS, callback=CheckpointCallback(save_freq=5000, save_path=log_dir,))
        model.save("trained")
    else:
        model = A2C.load("trained")
        


    # plot results
    # results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "Train")

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
    plt.show()
