import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from popgym.core.deck import Deck
from popgym.core.env import POPGymEnv


class Concentration(POPGymEnv):
    """Classic game of concentration. A deck of cards is shuffled and placed
    face-down. The player can flip two cards, if they match they get a reward
    otherwise they dont.

    Args:
        num_decks: Number of decks to play with
        deck_type: String denoting what we are matching. Can be the card colors
        (colors) or the card ranks (ranks)

    Returns:
        A gym environment
    """

    obs_requires_prev_action = True

    def __init__(self, num_decks=1, deck_type="ranks"):
        # See https://math.stackexchange.com/questions/1876467/
        # how-many-turns-on-average-does-it-take-for-a-perfect-player
        # -to-win-concentrati
        self.deck = Deck(num_decks=num_decks)
        n = 52 * num_decks
        self.episode_length = math.ceil(2 * n - (n / (2 * n - 1)))
        self.success_reward_scale = 1 / (self.deck.num_cards // 2)
        self.failure_reward_scale = -1 / (self.episode_length)

        if deck_type == "colors":
            self.deck_type = self.deck.colors
            self.deck_idx_type = self.deck.colors_idx
        # Cannot do suits because there are 13 of each suit -- we need
        # an even number to match
        elif deck_type == "ranks":
            self.deck_type = self.deck.ranks
            self.deck_idx_type = self.deck.ranks_idx
        else:
            raise NotImplementedError(f"Invalid deck type {deck_type}")

        # cards = 14 * np.ones(len(self.deck))
        self.facedown_card = len(np.unique(self.deck_type))
        cards = (1 + self.facedown_card) * np.ones(self.deck.num_cards)
        self.observation_space = gym.spaces.MultiDiscrete(cards)
        self.state_space = gym.spaces.Tuple(
            (
                gym.spaces.MultiDiscrete(cards - 1),  # cards values
                gym.spaces.MultiBinary(self.deck.num_cards),  # cards faces on
                gym.spaces.Discrete(n),  # first card turned
                gym.spaces.Discrete(n),  # second card turned
            )
        )
        self.action_space = gym.spaces.Discrete(self.deck.num_cards)
        self.deck.add_players("face_up", "face_up_idx", "in_play", "in_play_idx")
        self.last_in_play_idx = []
        self.last_trying_card_already_up = False

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        reward = 0
        terminated = truncated = False

        # Done conditions
        if self.curr_step >= self.episode_length - 1:
            truncated = True

        # Flip card
        self.deck["in_play"].append(self.deck.idx[action])
        self.deck["in_play_idx"].append(action)

        assert len(self.deck["in_play"]) <= 2
        # Cannot flip the same card twice
        self.obs = self.get_obs()
        self.last_in_play_idx = self.deck["in_play_idx"].copy()

        assert len(self.deck["face_up"]) == len(self.deck["face_up_idx"])
        assert len(self.deck["in_play"]) == len(self.deck["in_play_idx"])

        trying_card_already_up = any(
            idx in self.deck["face_up_idx"] for idx in self.deck["in_play_idx"]
        )

        # End of phase
        if trying_card_already_up:
            reward = self.failure_reward_scale * len(self.deck["in_play_idx"])
            self.deck.discard_hands("in_play", "in_play_idx")
        elif len(self.deck["in_play"]) == 2:
            inplay_cards = self.deck_type[self.deck["in_play"]]
            cards_match = inplay_cards[0] == inplay_cards[1]
            flipped_same_idx = self.deck["in_play"][0] == self.deck["in_play"][1]
            if cards_match and not flipped_same_idx:
                reward = self.success_reward_scale
                self.deck["face_up"].extend(self.deck["in_play"])
                self.deck["face_up_idx"].extend(self.deck["in_play_idx"])
                terminated = len(self.deck["face_up"]) == len(self.deck)
            else:
                reward = 2 * self.failure_reward_scale

            # Flip two last flipped-up cards face down again
            self.deck.discard_hands("in_play", "in_play_idx")
            self.hand: List[int] = []

        self.curr_step += 1

        return self.obs, reward, terminated, truncated, {}

    def render(self, mode="ascii"):
        self.obs.tolist()
        visible_mask = self.obs != self.facedown_card
        rend = np.full(self.obs.shape, "?")
        rend[visible_mask] = self.obs[visible_mask]
        rend = rend.reshape(4, 13)
        output = (
            " "
            + str(rend)
            .replace("[", "")
            .replace("]", "")
            .replace(",", "")
            .replace("'", "")
            # .replace("\n", "")
        )
        print(output)

    def get_obs(self):
        obs = self.facedown_card * np.ones(len(self.deck), dtype=np.int64)
        hand_idx = self.deck["face_up_idx"] + self.deck["in_play_idx"]
        deck_idx = self.deck.idx[hand_idx]
        obs[hand_idx] = self.deck_idx_type[deck_idx]
        return obs

    def get_state(self):
        cards_face_up = np.zeros(
            len(
                self.deck,
            ),
            dtype=np.int8,
        )
        cards_face_up[self.deck["face_up_idx"]] = 1
        first_card, second_card = self.facedown_card, self.facedown_card
        if len(self.last_in_play_idx) == 2:
            first_card, second_card = self.last_in_play_idx
        elif len(self.last_in_play_idx) == 1:
            first_card = self.last_in_play_idx[0]
        return self.state.copy(), cards_face_up, first_card, second_card

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[gym.core.ObsType, Dict[str, Any]]:
        super().reset(seed=seed)
        self.flip_next_turn = False
        self.deck.reset(rng=self.np_random)
        self.state = self.deck_idx_type[self.deck.idx].copy().astype(np.float32)
        self.curr_step = 0
        self.turn_counter = 0
        self.flipped_counter = 0
        self.last_in_play_idx = []
        self.last_trying_card_already_up = False
        self.obs = self.get_obs()
        info: Dict[str, Any] = {}
        return self.obs, info


class ConcentrationEasy(Concentration):
    def __init__(self):
        super().__init__(num_decks=1, deck_type="colors")


class ConcentrationMedium(Concentration):
    def __init__(self):
        super().__init__(num_decks=2, deck_type="colors")


class ConcentrationHard(Concentration):
    def __init__(self):
        super().__init__(num_decks=1, deck_type="ranks")
