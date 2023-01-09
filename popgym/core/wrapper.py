from typing import Optional

import gym
from gym import spaces
from gym.core import ObsType

from popgym.core.env import POPGymEnv


class POPGymWrapper(gym.Wrapper, POPGymEnv):

    def __init__(self, env: POPGymEnv):
        super().__init__(env)
        assert isinstance(env, POPGymEnv)
        self._state_space: Optional[spaces.Space] = None

    @property
    def state_space(self) -> spaces.Space:
        """Returns the observation space of the environment."""
        if self._state_space is None:
            return self.env.state_space
        return self._state_space

    @state_space.setter
    def state_space(self, space: spaces.Space):
        self._state_space = space

    def get_state(self) -> ObsType:
        return self.env.get_state()
