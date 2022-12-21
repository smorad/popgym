import copy
from typing import Callable, List

import gymnasium as gym
import numpy as np


def ascii_version_of_card(ranks, suits, return_string=True):
    """Instead of a boring text version of the card we render an ASCII image of
    the card.

    :param cards: One or more card objects
    :param return_string: By default we return the string version
        of the card, but the dealer hide the 1st card and we
    keep it as a list so that the dealer can add a hidden card in front of the list
    """
    # we will use this to prints the appropriate icons for each card
    suits_name = ["s", "d", "h", "c"]
    suits_symbols = ["♠", "♦", "♥", "♣"]

    # create an empty list of list, each sublist is a line
    lines = [[] for i in range(9)]

    for s, r in zip(suits, ranks):
        # "King" should be "K" and "10" should still be "10"
        if r == "10":  # ten is the only one who's rank is 2 char long
            rank = r
            space = ""  # if we write "10" on the card that line will be 1 char to long
        else:
            rank = r
            space = " "  # no "10", we use a blank space to will the void
        # get the cards suit in two steps
        suit = suits_name.index(s)
        suit = suits_symbols[suit]

        # add the individual card on a line by line basis
        lines[0].append("┌─────────┐")
        lines[1].append(
            "│{}{}       │".format(rank, space)
        )  # use two {} one for char, one for space or char
        lines[2].append("│         │")
        lines[3].append("│         │")
        lines[4].append("│    {}    │".format(suit))
        lines[5].append("│         │")
        lines[6].append("│         │")
        lines[7].append("│       {}{}│".format(space, rank))
        lines[8].append("└─────────┘")

    result = []
    for index, line in enumerate(lines):
        result.append("".join(lines[index]))

    # hidden cards do not use string
    if return_string:
        return "\n".join(result)
    else:
        return result


class DeckEmptyError(Exception):
    pass


RANKS = np.array(["a", "2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k"])
SUITS = np.array(["s", "d", "c", "h"])
SUITS_UNICODE = ["♠", "♦", "♥", "♣"]
COLORS = np.array(["b", "r"])
DECK_SIZE = 52


class Deck:
    """An object that represents a collection of cards.

    A deck can represent a single deck or multiple decks
    """

    def get_obs_space(self, fields=["colors", "suits", "ranks"], hand_size=1):
        space = []
        for f in fields:
            if f == "colors":
                space.append(COLORS.size)
            elif f == "suits":
                space.append(SUITS.size)
            elif f == "ranks":
                space.append(RANKS.size)
            else:
                raise Exception(f"Invalid field: {f}")
        space = np.tile(np.array(space), hand_size)
        if len(space) == 1:
            return gym.spaces.Discrete(space[0])
        return gym.spaces.MultiDiscrete(space)

    def __init__(self, num_decks=1, shuffle=False):
        self.num_decks = num_decks
        self.num_cards = DECK_SIZE * num_decks
        self.idx = np.arange(self.num_cards)

        self.ranks = np.tile(RANKS.repeat(SUITS.size), num_decks)
        self.ranks_idx = np.tile(np.arange(RANKS.size).repeat(SUITS.size), num_decks)

        self.suits = np.tile(np.tile(SUITS, RANKS.size), num_decks)
        self.suits_idx = np.tile(np.tile(np.arange(SUITS.size), RANKS.size), num_decks)

        self.colors = np.tile(COLORS, self.num_cards // 2)
        self.colors_idx = np.tile(np.arange(COLORS.size), self.num_cards // 2)
        self.hands = {}
        # The length of the deck, which decreases as cards
        # are drawn/dealt
        self.deck_len = self.num_cards
        if shuffle:
            # For determinism, we should always shuffle from the original input
            self.idx = np.arange(self.num_cards)
            np.random.shuffle(self.idx)

        assert self.idx.size == self.num_cards
        assert self.ranks.size == self.num_cards
        assert self.suits.size == self.num_cards
        assert self.colors.size == self.num_cards
        assert self.ranks_idx.size == self.num_cards
        assert self.suits_idx.size == self.num_cards
        assert self.colors_idx.size == self.num_cards

        self.keys = ["idx", "ranks", "suits", "colors"]
        self.idx_keys = ["ranks_idx", "suits_idx", "colors_idx", "idx"]

    def define_hand_value(
        self, fn: Callable[[List[str]], int], fields: List[str]
    ) -> None:
        """Pass in a function to be used to define the value of a hand
        or set of cards"""
        self.value_fn = fn
        for f in fields:
            assert f in self.keys, f"{f} is not {self.keys}"
        self.value_fn_args = fields

    def clone(self) -> "Deck":
        return copy.deepcopy(self)

    def value(self, player: str) -> int:
        """Returns the value of a players hand by calling the function passed
        to define_hand_value"""
        assert hasattr(
            self, "value_fn"
        ), "Must specify a value fn using define_hand_value first!"

        args = []
        for arg in self.value_fn_args:
            idx = self.hands[player]
            args.append(getattr(self, arg)[idx])

        # Numpy will attempt to unpack an ndarray
        # if it is the only element in a list
        # when using a starred expression
        # i.e. len([np.array([1,2,3]])) == 3
        if len(args) == 1:
            return self.value_fn(args[0])
        else:
            return self.value_fn(*args)

    def value_idx(self, idx: List[int]) -> int:
        """Returns the value of a selection of cards"""
        assert hasattr(
            self, "value_fn"
        ), "Must specify a value fn using define_hand_value first!"
        args = []
        for arg in self.value_fn_args:
            args.append(getattr(self, arg)[idx])

        return self.value_fn(*args)

    def __len__(self):
        return self.deck_len

    def __getitem__(self, item):
        return self.hands[item]

    def add_players(self, *players: List[str]) -> None:
        for player in players:
            self.hands[player] = []

    def deal(self, player: str, num_cards: int = 1) -> None:
        """Deals a number of cards to the specified player
        from the deck"""
        new_len = self.deck_len - num_cards
        if new_len < 0:
            raise DeckEmptyError()

        self.hands[player] += self.idx[new_len : self.deck_len].tolist()
        self.deck_len = new_len

    def discard_hands(self, *players: List[str]):
        """Discards the cards in the hand of a player. Note that
        these cards do not go back into the deck. Call reset()
        to fold the hands back into the deck"""
        for player in players:
            self.hands[player].clear()

    def discard_all(self):
        """Discards the cards in all player hands. Note that
        these cards do not go back into the deck. Call reset()
        to fold the hands back into the deck"""
        for player in self.hands:
            self.hands[player].clear()

    def discard(self, player: str, hand_idx: int):
        """Discards one card in the players hand at the specified idx.
        Note this idx refers to the idx of the card in the hand, rather
        than the idx of the card in the deck"""
        self.hands[player].pop(hand_idx)

    def reset(self, shuffle=True, rng=None):
        """Empties the hands of all players and places cards
        back into the deck in their original position. Optionally
        shuffles the deck afterwards"""
        for hands in self.hands.values():
            hands.clear()
        self.deck_len = self.num_cards
        if shuffle:
            self.idx = np.arange(self.num_cards)
            if rng is None:
                np.random.shuffle(self.idx)
            else:
                self.rng = rng
                rng.shuffle(self.idx)

    def show(
        self, player: str, fields: List[str] = ["colors", "suits", "ranks"], pad_to=None
    ) -> List[np.ndarray]:
        """Shows the hand of the player, returning the fields specified of the cards
        they hold. Optionally zero-pad to a size."""
        reprs = []
        if pad_to is not None:
            padding = [0] * (pad_to - len(self.hands[player]))
        else:
            padding = []
        hand_idx = np.array(self.hands[player] + padding, dtype=np.int64)
        for f in fields:
            assert f in [*self.idx_keys, *self.keys], f"{f} is not a valid key"
            arr = getattr(self, f)
            # Special case, do not double index indices
            if f == "idx":
                reprs.append(np.array(hand_idx))
            else:
                # Requires indexing
                reprs.append(arr[hand_idx])

        return np.stack(reprs)

    def hand_size(self, player: str) -> int:
        return len(self.hands[player])

    def visualize(self, player: str) -> str:
        """Returns a string visualization of a player's hand, for printing
        to the terminal"""
        if len(self[player]) == 0:
            return "\n".join([""] * 10)
        ranks, suits = self.show(player, ["ranks", "suits"])
        return ascii_version_of_card(ranks, suits)

    def visualize_idx(self, idx: List[int]) -> str:
        """Returns a string visualization of the following idx,
        referring to cards in the hand or deck"""
        if len(idx) == 0:
            return "\n".join([""] * 10)
        suits = self.suits[idx]
        ranks = self.ranks[idx]
        return ascii_version_of_card(ranks, suits)
