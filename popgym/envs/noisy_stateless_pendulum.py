from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from popgym.envs.stateless_pendulum import StatelessPendulum


class NoisyStatelessPendulum(StatelessPendulum):
    def __init__(self, *args, **kwargs):
        noise_sigma = kwargs.pop("noise_sigma", 0.1)
        super().__init__(*args, **kwargs)
        self.noise_sigma = np.full(self.observation_space.shape, noise_sigma)

    def add_noise_to_obs(self, obs):
        noise = self.np_random.normal(0, self.noise_sigma).astype(np.float32)
        obs_space = self.observation_space
        obs = np.clip(obs + noise, obs_space.low, obs_space.high)
        return obs

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        next_obs, reward, terminated, truncated, info = super().step(action)
        return self.add_noise_to_obs(next_obs), reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[gym.core.ObsType, dict]:

        init_obs, info = super().reset(seed=seed, options=options)
        init_obs = self.add_noise_to_obs(init_obs)
        return init_obs, info


class NoisyStatelessPendulumEasy(NoisyStatelessPendulum):
    def __init__(self):
        super().__init__(noise_sigma=0.1)


class NoisyStatelessPendulumMedium(NoisyStatelessPendulum):
    def __init__(self):
        super().__init__(noise_sigma=0.2)


class NoisyStatelessPendulumHard(NoisyStatelessPendulum):
    def __init__(self):
        super().__init__(noise_sigma=0.3)
