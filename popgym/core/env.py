from abc import abstractmethod

from gymnasium import Env, spaces
from gymnasium.core import ObsType

from popgym.core.observability import Observability


class POPGymEnv(Env):
    """A wrapper around gym.Env providing utilities for partial observability"""

    # The gym space for the underlying hidden Markov state
    state_space: spaces.Space[ObsType]
    # The observability level determines if and how the hidden Markov state
    # is returned from step() via the ObservabilityWrapper
    observability_level: Observability = Observability.PARTIAL
    # Whether the observation must contain the
    # previous action to learn the optimal policy.
    # If true, consider using the LastActionWrapper
    obs_requires_prev_action: bool = False

    @abstractmethod
    def get_state(self) -> ObsType:
        """Returns the underlying hidden Markov state"""
