

import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space['camera'].shape[:2]
        self.observation_space['camera'] = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        camera = observation['camera']

        # permute [H, W, C] array to [C, H, W] tensor
        camera = np.transpose(camera, (2, 0, 1))
        camera = torch.tensor(camera.copy(), dtype=torch.uint8)

        observation['camera'] = camera
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)

        camera = observation['camera']
        transform = T.Grayscale()
        camera = transform(camera)
        observation['camera'] = camera
        return observation

