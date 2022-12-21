from typing import Any, Dict, Tuple

from gymnasium import Env, Wrapper, spaces
from gymnasium.core import ObsType

from popgym.core.env import POPGymEnv
from popgym.core.observability import OBS, STATE, Observability


class Markovian(Wrapper):
    """Wrapper that adds the hidden Markov state to the observation or info dict

    Args:
        env: The environment.
        observability: The observability level.

    Returns:
        A gym environment
    """

    def __init__(self, env: Env, observability: Observability):
        super(Markovian, self).__init__(env)
        assert isinstance(
            env.unwrapped, POPGymEnv
        ), "This wrapper is made for POPGymEnvs."
        self.observability = observability
        if observability == Observability.FULL_AND_PARTIAL:
            self.observation_space = spaces.Dict(
                {
                    OBS: self.env.observation_space,
                    STATE: self.state_space,
                }
            )
        elif observability == Observability.FULL:
            self.observation_space = self.state_space
        elif observability in [Observability.PARTIAL, Observability.FULL_IN_INFO_DICT]:
            pass
        else:
            raise NotImplementedError("Invalid observability level:", observability)

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[Any, Any]]:
        obs, info = self.env.reset(**kwargs)
        obs, info = self.add_state(obs, info)
        return obs, info

    def add_state(self, obs, info):
        if self.observability == Observability.FULL:
            obs = self.get_state()
        elif self.observability == Observability.FULL_AND_PARTIAL:
            obs = {OBS: obs, STATE: self.get_state()}
        elif self.observability == Observability.FULL_IN_INFO_DICT:
            info[STATE] = self.get_state()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs, info = self.add_state(obs, info)
        return obs, reward, terminated, truncated, info
