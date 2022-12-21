from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from popgym.core.deck import RANKS, Deck
from popgym.core.env import POPGymEnv


def value_fn(hand):
    if hand[-1] > hand[-2]:
        return 1
    elif hand[-1] == hand[-2]:
        return 0
    else:
        return -1


class HigherLower(POPGymEnv):
    """A game of higher/lower. Given a deck of cards, the agent predicts whether the
    next card drawn from the deck is higher or lower than the last card drawn from
    the deck. A push results in zero reward, while a correct/incorrect guess result
    in 1/deck_size and -1/deck_size reward. The agent can learn to count cards to
    infer which cards are left in the deck, improving accuracy.

    Args:
        num_decks: The number of individual decks combined into a single deck.

    Returns:
        A gym environment
    """

    def __init__(self, num_decks=1):
        self.num_decks = num_decks
        self.deck = Deck(num_decks)
        self.deck.add_players("player")
        self.action_space = gym.spaces.Discrete(2)
        self.state = np.zeros(
            (
                len(
                    RANKS,
                )
            ),
            dtype=np.uint8,
        )
        self.observation_space = self.deck.get_obs_space(["ranks"])
        self.state_space = gym.spaces.Box(0, 1, self.state.shape)
        self.value_map = dict(zip(self.deck.ranks, range(len(self.deck.ranks))))
        self.deck_size = len(self.deck)

    def get_state(self):
        return (self.state.copy() / 4 / self.num_decks).astype(np.float32)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        guess_higher = action == 0
        terminated = len(self.deck) <= 1

        self.deck.deal("player", 1)
        assert self.deck.hand_size("player") == 2
        curr_value, next_value = self.deck.show("player", ["ranks_idx"]).reshape(-1)
        self.state[next_value] += 1

        rew_scale = 1 / self.deck_size
        if next_value == curr_value:
            reward = 0
        elif (next_value > curr_value) == guess_higher:
            reward = rew_scale
        else:
            reward = -rew_scale

        self.deck.discard("player", 0)
        obs = self.deck.show("player", ["ranks_idx"]).item()

        return obs, reward, terminated, False, {}

    def render(self, mode="ascii"):
        return self.deck.visualize_idx(self.deck["player"])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[gym.core.ObsType, Dict[str, Any]]:
        super().reset(seed=seed)
        self.deck.reset(rng=self.np_random)
        self.deck.deal("player", 1)
        self.state = np.zeros(
            (
                len(
                    RANKS,
                )
            ),
            dtype=np.uint8,
        )
        obs = self.deck.show("player", ["ranks_idx"]).item()
        self.state[obs] = 1
        viz = np.concatenate(self.deck.show("player", ["suits", "ranks"]))
        return obs, {"card": viz}


class HigherLowerEasy(HigherLower):
    def __init__(self):
        super().__init__(num_decks=1)


class HigherLowerMedium(HigherLower):
    def __init__(self):
        super().__init__(num_decks=2)


class HigherLowerHard(HigherLower):
    def __init__(self):
        super().__init__(num_decks=3)
