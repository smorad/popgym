from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from popgym.core.deck import Deck
from popgym.core.env import POPGymEnv


class CountRecall(POPGymEnv):
    """A game where the agent is queried on how many times it has observed
    an event in the past. This tests long-term order-agnostic memory like sets.

    Args:
        max_episode_length: The maximum number of timesteps in an episode
        error_clamp: Denotes the domain of the linear portion of the reward
            function. E.g. error_clamp == 2 means the agent will receive
            linearly-decreasing rewards for counts off by up to 2. Errors greater
            than 2 provide the same negative reward as an error of 2. Note as
            error_clamp -> inf, a crappy agent and perfect agent will acheive
            the same total reward.
        deck_type: What we use to count/differentiate cards.
            Can be colors, suits, or ranks.

    Returns:
        A gym environment
    """

    query: Optional[int]

    def __init__(self, num_decks=1, error_clamp=0.5, deck_type="colors"):
        self.value_deck = Deck(num_decks=num_decks)
        self.query_deck = Deck(num_decks=num_decks)
        self.value_deck.add_players("in_play")
        self.query_deck.add_players("in_play")
        if deck_type == "colors":
            self.deck_type = self.value_deck.colors
            self.deck_idx_type = self.value_deck.colors_idx
        elif deck_type == "suits":
            self.deck_type = self.value_deck.suits
            self.deck_idx_type = self.value_deck.suits_idx
        elif deck_type == "ranks":
            self.deck_type = self.value_deck.ranks
            self.deck_idx_type = self.value_deck.ranks_idx
        else:
            raise NotImplementedError(f"Invalid deck type {deck_type}")

        self.num_distinct_cards = len(np.unique(self.deck_type))
        self.max_card_count = int(self.value_deck.num_cards / self.num_distinct_cards)
        # Space: [dealt card, card query]
        self.observation_space = gym.spaces.MultiDiscrete([self.num_distinct_cards] * 2)
        self.state_space = gym.spaces.Tuple(
            (
                gym.spaces.Box(0.0, 1.0, (self.num_distinct_cards,)),
                gym.spaces.Box(0.0, 1.0, (self.num_distinct_cards,)),
                # The current query is needed, but not the dealt card
                self.observation_space,
            )
        )
        self.last_obs = np.array([0, 0])

        self.action_space = gym.spaces.Discrete(self.max_card_count)
        self.max_episode_length = self.value_deck.num_cards - 1
        self.error_clamp = error_clamp
        self.reward_scale = 1 / self.max_episode_length

    def render(self):
        dealt = self.deck_type[self.value]
        queried = self.deck_type[self.query]
        print(f"Dealt {dealt}, recall {queried} count")

    def get_state(self):
        state = (
            (self.counts / self.max_card_count).astype(np.float32),
            (self.query_counts / self.max_card_count).astype(np.float32),
            self.last_obs.copy(),
        )
        return state

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        if isinstance(action, np.ndarray):
            action = action.item()

        self.prev_query = self.query
        prev_count = self.counts[self.prev_query]
        self.value, self.query = self.deal()
        self.counts[self.value] += 1
        self.query_counts[self.query] += 1

        reward = 1 if action == prev_count else -1
        reward *= self.reward_scale

        terminated = len(self.value_deck) == 0

        obs = np.array([self.value, self.query], dtype=np.int64)
        self.last_obs = obs
        info = {"counts": self.counts}

        return obs.copy(), reward, terminated, False, info

    def sample_deck(self):
        return np.random.choice(self.deck_idx_type, size=self.max_episode_length + 1)

    def deal(self):
        self.value_deck.deal("in_play")
        self.query_deck.deal("in_play")
        return (
            self.deck_idx_type[self.value_deck["in_play"][-1]],
            self.deck_idx_type[self.query_deck["in_play"][-1]],
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[gym.core.ObsType, Dict[str, Any]]:
        super().reset(seed=seed)
        self.value_deck.reset(rng=self.np_random)
        self.query_deck.reset(rng=self.np_random)  # Having another PRNG is not needed.

        self.counts = np.zeros((self.num_distinct_cards,), dtype=np.int64)
        self.query_counts = np.zeros((self.num_distinct_cards,), dtype=np.int64)

        self.value, self.query = self.deal()
        self.prev_query = None
        self.counts[self.value] += 1
        self.query_counts[self.query] += 1

        obs = np.array([self.value, self.query], dtype=np.int64)
        self.last_obs = obs.copy()
        return obs, {"counts": self.counts}


class CountRecallEasy(CountRecall):
    pass


class CountRecallMedium(CountRecall):
    def __init__(self, *args, **kwargs):
        super().__init__(num_decks=2, deck_type="suits")


class CountRecallHard(CountRecall):
    def __init__(self, *args, **kwargs):
        super().__init__(num_decks=4, deck_type="ranks")
