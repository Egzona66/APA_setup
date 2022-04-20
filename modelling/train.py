
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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common import results_plotter

import torch as th
from rich.progress import track

from modelling.utils import grab_frames, setup_video
from modelling.environment import RLEnvironment
from modelling.networks import ProprioceptiveEncoder, VisualProprioceptionCombinedExtractor, CustomActorCriticPolicy
from modelling.wrappers import SkipFrame, GrayScaleObservation, FrameStack

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

        # env = SkipFrame(env, skip=4)
        env = GrayScaleObservation(env)
        return env

    set_random_seed(seed)
    return _init 



def make_agent(env, tensorboard_log=None):
    policy_kwargs = dict(
        features_extractor_class=VisualProprioceptionCombinedExtractor,
        # features_extractor_kwargs=dict(features_dim=16),
        activation_fn=th.nn.Tanh,
        net_arch=[256, 256]
    )


    model = A2C(
                CustomActorCriticPolicy,  # 'MultiInputPolicy', 
                env,   
                policy_kwargs=policy_kwargs, 
                learning_rate = 4e-3,
                n_steps = 8,
                gamma = 0.99,
                gae_lambda = .9,
                ent_coef = 0.0,
                vf_coef = 0.5,
                max_grad_norm = 0.5,
                rms_prop_eps = 1e-5,
                use_rms_prop = True,
                use_sde = True,
                sde_sample_freq = -1,
                normalize_advantage = False,
                device = "auto",
                verbose=1, 
                tensorboard_log=tensorboard_log,
    )

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
    NAME = "overnight_speed"
    N_CPU = 3
    N_STEPS = 800_000

    log_dir = f"./logs/{NAME}"
    os.makedirs(log_dir, exist_ok=True)

    # Create the vectorized environment
    if N_CPU > 1:
        env = DummyVecEnv([make_env(i, log_dir=log_dir) for i in range(N_CPU)])  # or SubprocVecEnv/DummyVecEnv
    else:
        env = Monitor(RLEnvironment(), log_dir)
        check_env(env)
    
    model = make_agent(env, tensorboard_log=log_dir)
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=log_dir,
                                    name_prefix=NAME)
    model.learn(total_timesteps=N_STEPS, callback=checkpoint_callback)
    model.save(f"trained_{NAME}")

    # make video
    video_length = 100
    video_name = os.path.join(log_dir, f"video_{NAME}.mp4")
    env = RLEnvironment()
    video = setup_video(video_name, env.env)

    # Record the video starting at the first step
    model = A2C.load(f"trained_{NAME}")

    obs = env.reset()
    for i in track(range(video_length + 1), description="Recording video", total=video_length + 1):
        action = model.predict(obs)
        obs, _, _, _ = env.step(action)

        frame = grab_frames(env.env)
        video.write(frame)

    # Save the video
    video.release()
    env.close()
    print("Done & saved video at ", video_name)
    plt.show()
