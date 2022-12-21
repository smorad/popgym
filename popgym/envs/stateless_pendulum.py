# Inspired by ray rllib at
# https://github.com/ray-project/ray/blob/master/rllib/examples/env/stateless_pendulum.py

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.envs.classic_control import PendulumEnv
from gymnasium.spaces import Box

from popgym.core.env import POPGymEnv


class StatelessPendulum(PendulumEnv, POPGymEnv):
    """Partially observable variant of the Pendulum gym environment.
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/
    pendulum.py
    We delete the angular velocity component of the state, so that it
    can only be solved by a memory enhanced model (policy).

    Args:
        max_episode_length: Exactly what it sounds like
    Returns:
        A gym environment
    """

    def __init__(self, *args, **kwargs):
        self.max_episode_length = kwargs.pop("max_episode_length", 200)
        super().__init__(*args, **kwargs)

        # Fix our observation-space (remove angular velocity component).
        high = np.array([1.0, 1.0], dtype=np.float32)
        self.state_space = self.observation_space
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

    def reward_transform(self, reward):
        low, high = -16.2736044, 0
        shifted = reward + (high - low) / 2
        scaled = shifted / ((high - low) / 2)
        transformed = scaled / self.max_episode_length
        return transformed

    def get_state(self):
        return self._state

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        next_obs, reward, terminated, truncated, info = super().step(action)
        self._state = next_obs
        self.num_steps += 1
        if self.num_steps >= self.max_episode_length:
            truncated = True
        reward = self.reward_transform(reward)
        # next_obs is [cos(theta), sin(theta), theta-dot (angular velocity)]
        return next_obs[:-1], reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[gym.core.ObsType, dict]:
        self.num_steps = 0
        init_obs, info = super().reset(seed=seed, options=options)
        self._state = init_obs
        return init_obs[:-1], info


class StatelessPendulumEasy(StatelessPendulum):
    pass


class StatelessPendulumMedium(StatelessPendulum):
    def __init__(self):
        super().__init__(max_episode_length=150)


class StatelessPendulumHard(StatelessPendulum):
    def __init__(self):
        super().__init__(max_episode_length=100)
