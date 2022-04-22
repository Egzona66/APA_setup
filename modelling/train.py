
import sys

sys.path.append("./")

from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from stable_baselines3 import A2C, DDPG
from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage, VecVideoRecorder
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import torch as th
from rich.progress import track

from modelling.utils import grab_frames, setup_video
from modelling.environment import RLEnvironment
from modelling.networks import VisualProprioceptionCombinedExtractor, CustomActorCriticPolicy, ProprioceptiveFeaturesExtractor
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
        # env = GrayScaleObservation(env)
        # env = VecVideoRecorder(env, log_dir,
        #                record_video_trigger=lambda x: x == 0, 
        #                video_length=100,
        #                name_prefix="vid")
        return env

    set_random_seed(seed)
    return _init 



def make_agent(env, tensorboard_log=None, params=dict()):
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))



    policy_kwargs = dict(
        # features_extractor_class=VisualProprioceptionCombinedExtractor,
        features_extractor_class=ProprioceptiveFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=32),
        net_arch=[512, 512, 512]
    )

    # model = DDPG(
    #     "MultiInputPolicy",
    #     env,
    #     policy_kwargs=policy_kwargs,
    #     tensorboard_log=tensorboard_log,
    #     action_noise=action_noise,
    #     **params
    # )


    model = A2C(
                # CustomActorCriticPolicy,  # 'MultiInputPolicy', 
                'MlpPolicy',
                env,   
                policy_kwargs=policy_kwargs, 
                tensorboard_log=tensorboard_log,
                **params,
    )

    # model = RecurrentPPO(
    #         "MlpLstmPolicy",
    #         env=env,
    #         policy_kwargs=policy_kwargs,
    #         tensorboard_log=tensorboard_log,
    #         **params,
    # )
    return model



# TODO check if env.reset is broken


a2c_params = dict(
                learning_rate = 0.003,
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
    )

ddpo_params = dict(
    learning_rate = 1e-3,
    gamma = 0.99,
    learning_starts = 10000,
    # noise_type = 'normal',
    # noise_std = 0.1,
    verbose=1,
    device = "auto",
)

# TODO try TD3 & PPO
# TODO try with action noise

if __name__ == '__main__':
    NAME = "newstart"
    N_CPU = 2
    N_STEPS = 800_000
    SEED = 0

    PARAMS = a2c_params

    log_dir = Path(f"./logs/{NAME}")
    log_dir.mkdir(exist_ok=True)

    # get subdirs in log_dir to save in a separate one
    subdirs = [x for x in log_dir.iterdir() if x.is_dir()]
    log_dir = str(log_dir / ("rep_"+ str(len(subdirs)+1)))
    os.makedirs(log_dir, exist_ok=True)

    # save params to json
    with open(os.path.join(log_dir, 'data.json'), 'w') as fp:
        prms = {**PARAMS, **dict(ncpu=N_CPU, nsteps=N_STEPS, seed=SEED)}
        json.dump(prms, fp, indent=4)

    # Create the vectorized environment
    if N_CPU > 1:
        env = DummyVecEnv([make_env(i, log_dir=log_dir, seed=SEED) for i in range(N_CPU)])  # or SubprocVecEnv/DummyVecEnv
    else:
        env = make_env(1, seed=SEED)()
        check_env(env)
    
    model = make_agent(env, tensorboard_log=log_dir, params=PARAMS)
    checkpoint_callback = CheckpointCallback(save_freq=2500, save_path=log_dir,
                                    name_prefix=NAME)
    model.learn(total_timesteps=N_STEPS, callback=checkpoint_callback, log_interval=10)
    model.save(f"trained_{NAME}")

    # make video
    video_length = 100
    video_name = os.path.join(log_dir, f"video_{NAME}.mp4")
    env = make_env(1, seed=SEED)()
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
    print("Done & saved video at ", video_name)
    plt.show()
