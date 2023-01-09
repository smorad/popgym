from typing import Optional, Tuple

import numpy as np
from gym import spaces
from gym.core import ActType, ObsType

from popgym.core.observability import OBS, STATE
from popgym.core.wrapper import POPGymWrapper
from popgym.core.env import POPGymEnv

PREV_ACTION = "prev_action"


class PreviousAction(POPGymWrapper):
    """Wrapper that adds the last action to the observation.

    Args:
        env: The environment
        null_action: Optional null action that is returned when resetting the
            environment. If not provided, the null action will be 0
            (int or vector) if it is in the action space, or the lowest action
            possible.

    Returns:
        A gym environment
    """

    def __init__(self, env: POPGymEnv, null_action: Optional[ActType] = None):
        super().__init__(env)
        self.observation_space = PreviousAction.add_act_space_to_obs_space(
            self.env.observation_space, self.env.action_space
        )
        if null_action is None:
            null_action = PreviousAction.get_null_action(self.action_space)
        assert self.action_space.contains(null_action)
        self.null_action = null_action

    @staticmethod
    def add_act_space_to_obs_space(
        observation_space: spaces.Space, action_space: spaces.Space
    ) -> spaces.Space:
        """
        Returns a modified observation space to account for the last action.
        Args:
            observation_space: Original observation space
            action_space: Action space

        Returns:
            The new observation space
        """
        if isinstance(
            observation_space,
            (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        ):
            observation_space = spaces.Tuple((observation_space, action_space))
        elif isinstance(observation_space, spaces.Tuple):
            observation_space = spaces.Tuple(
                tuple(observation_space.spaces) + (action_space,)
            )
        elif isinstance(observation_space, spaces.Dict):
            observation_space = observation_space.spaces.copy()
            if set(observation_space.keys()) == {OBS, STATE}:
                # Observation comes from ObservabilityWrapper with
                # observability level FULL_AND_PARTIAL
                observation_space[OBS] = PreviousAction.add_act_space_to_obs_space(
                    observation_space[OBS], action_space
                )
            else:
                observation_space[PREV_ACTION] = action_space
            observation_space = spaces.Dict(observation_space)
        else:
            raise NotImplementedError("Unknown observation space")
        return observation_space

    @staticmethod
    def add_act_to_obs(
        observation_space: spaces.Space, obs: ObsType, action: ActType
    ) -> ObsType:
        """
        Static method that adds the action to the observation.
        Args:
            observation_space: Original observation space of the environment.
            obs: The observation.
            action: The action.

        Returns:
            Modified observation.
        """
        if isinstance(
            observation_space,
            (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
        ):
            obs = (obs, action)
        elif isinstance(observation_space, spaces.Tuple):
            obs = (*obs, action)
        elif isinstance(observation_space, spaces.Dict):
            if set(observation_space.keys()) == {OBS, STATE}:
                # Observation comes from ObservabilityWrapper with
                # observability level FULL_AND_PARTIAL
                obs[OBS] = PreviousAction.add_act_to_obs(
                    observation_space[OBS], obs[OBS], action
                )
            else:
                obs[PREV_ACTION] = action
        else:
            raise NotImplementedError("Unknown observation space")
        return obs

    @staticmethod
    def get_null_action(action_space: spaces.Space) -> ActType:
        """
        Static method that generates a null action based on the action space.
        Args:
            action_space: The action space.

        Returns:
            The null action.
        """
        if isinstance(
            action_space,
            (spaces.Discrete, spaces.MultiBinary, spaces.MultiDiscrete, spaces.Box),
        ):
            action = np.zeros(action_space.shape, action_space.dtype)
            if not action_space.contains(action):
                action = action_space.low
        elif isinstance(action_space, spaces.Tuple):
            action = tuple(
                PreviousAction.get_null_action(action_space_)
                for action_space_ in action_space
            )
        elif isinstance(action_space, spaces.Dict):
            action = {
                key: PreviousAction.get_null_action(value)
                for key, value in action_space.items()
            }
        else:
            raise NotImplementedError
        return action

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        obs = PreviousAction.add_act_to_obs(self.env.observation_space, obs, action)
        return obs, reward, done, info

    def reset(self, **kwargs):
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
            obs = PreviousAction.add_act_to_obs(
                self.env.observation_space, obs, self.null_action
            )
            return obs, info
        else:
            obs = self.env.reset(**kwargs)
            obs = PreviousAction.add_act_to_obs(
                self.env.observation_space, obs, self.null_action
            )
            return obs
