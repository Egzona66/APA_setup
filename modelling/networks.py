import gym
import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ProprioceptiveEncoder(BaseFeaturesExtractor):
    """
    A multi-layer perceptron NN for proprioceptive observations (e.g. joint angles)
    representation.

    See: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    And: https://github.com/christianversloot/machine-learning-articles/blob/main/creating-a-multilayer-perceptron-with-pytorch-and-lightning.md

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(ProprioceptiveEncoder, self).__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, features_dim)
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.Tanh())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))