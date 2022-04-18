
import sys
sys.path.append("./")


import matplotlib.pyplot as plt
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter

from rich.progress import track

from modelling.utils import grab_frames, setup_video
from modelling.environment import RLEnvironment



def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RLEnvironment()
        # you can use this to test the env is correct: check_env(env)

        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True

if __name__ == '__main__':
    TRAIN = False

    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    if TRAIN:
        checkpoint_callback = CheckpointCallback(save_freq=2500, save_path=log_dir,
                                         name_prefix='rl_model')

        num_cpu = 10  # Number of processes to use
        # Create the vectorized environment
        # env = DummyVecEnv([Monitor(make_env(i, ), log_dir, ) for i in range(num_cpu)])  # or SubprocVecEnv/DummyVecEnv
        env = Monitor(RLEnvironment(), log_dir)

        # env = Monitor(env, log_dir)

        model = A2C('MultiInputPolicy', env, verbose=1)
        model.learn(total_timesteps=10_000, callback=checkpoint_callback)
        model.save("trained")
    else:
        model = A2C.load("trained")
        


    # plot results
    results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "Test")
    plot_results(log_dir)

    # make video
    video_length = 100
    video_name = "video.mp4"
    env = RLEnvironment()
    video = setup_video(video_name, env.env.env)

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