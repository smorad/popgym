import enum
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from popgym.core.env import POPGymEnv


class HiddenSquare(enum.IntEnum):
    CLEAR = 0
    MINE = 1
    VIEWED = 2


class MineSweeper(POPGymEnv):
    """Classic Microsoft MineSweeper but with the board obscured. Mines are hidden
    underneath a grid of tiles. The player clicks a tile, which returns the
    coordinates of the tile and how many mines are present in adjacent tiles.
    Clicking a mine loses. Player must click all free squares to win.

    Args:
        difficulty: Easy, medium, or hard. Sets the board size and number of
            mines based on difficulty.

    Returns:
        A gym environment
    """

    obs_requires_prev_action = True

    def __init__(self, difficulty="easy"):
        assert difficulty in ["easy", "medium", "hard"]
        if difficulty == "easy":
            dims = 4, 4
            num_mines = 2
        elif difficulty == "medium":
            dims = 6, 6
            num_mines = 6
        elif difficulty == "hard":
            dims = 8, 8
            num_mines = 10
        else:
            raise NotImplementedError(f"Invalid difficulty {difficulty}")

        self.dims = dims
        self.num_mines = num_mines
        self.max_episode_length = dims[0] * dims[1] - num_mines
        self.success_reward_scale = 1 / self.max_episode_length
        self.fail_reward_scale = -0.5 - self.success_reward_scale
        # -1 for one step less (last action must be mine for lowest G)
        # -1 for one step less (first action is a view which will give reward)
        self.bad_action_reward_scale = -0.5 / (self.max_episode_length - 2)
        self.observation_space = gym.spaces.Discrete(min(num_mines + 1, 10))
        self.state_space = gym.spaces.MultiDiscrete([3] * np.prod(self.dims))
        self.action_space = gym.spaces.MultiDiscrete(np.array(dims))

    def get_state(self):
        return self.hidden_grid.flatten().copy()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        terminated = truncated = False
        action = tuple(action)
        if self.hidden_grid[action] == HiddenSquare.MINE:
            terminated = True
            reward = self.fail_reward_scale
        elif self.hidden_grid[action] == HiddenSquare.VIEWED:
            # Querying already viewed square
            reward = self.bad_action_reward_scale
        else:
            self.hidden_grid[action] = HiddenSquare.VIEWED
            reward = self.success_reward_scale

        truncated = self.timestep == self.max_episode_length
        terminated |= np.all(self.hidden_grid != HiddenSquare.CLEAR).item()

        obs = self.neighbor_grid[action].item()
        self.timestep += 1

        return obs, reward, terminated, truncated, {}

    def render(self):
        visible_mask = self.hidden_grid == HiddenSquare.VIEWED
        result = np.full(self.hidden_grid.shape, ".")
        result[visible_mask] = self.neighbor_grid[visible_mask]
        out = " " + str(result).replace("[", "").replace("]", "").replace("'", "")
        print(out)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[gym.core.ObsType, Dict[str, Any]]:
        super().reset(seed=seed)
        # Init grids
        self.hidden_grid = np.full(self.dims, HiddenSquare.CLEAR, dtype=np.int8)
        mines_flat = self.np_random.choice(
            np.arange(self.dims[0] * self.dims[1]), size=self.num_mines, replace=False
        )
        self.hidden_grid.ravel()[mines_flat] = HiddenSquare.MINE
        self.neighbor_grid = np.zeros_like(self.hidden_grid)
        src_grid = np.pad(self.hidden_grid, [(1, 1), (1, 1)], constant_values=0)
        for shift_i in [-1, 0, 1]:
            for shift_j in [-1, 0, 1]:
                self.neighbor_grid += np.roll(
                    np.roll(src_grid, shift_i, 0), shift_j, 1
                )[1:-1, 1:-1]

        self.timestep = 0
        obs = 0
        return obs, {}


if __name__ == "__main__":
    e = MineSweeper()
    obs = e.reset()
    e.render()
    terminated = truncated = False
    while not terminated or truncated:
        action = np.array(input("Enter x,y:").split(",")).astype(np.int8)
        obs, reward, terminated, truncated, info = e.step(action)
        e.render()
        print("reward", reward)


class MineSweeperEasy(MineSweeper):
    def __init__(self):
        super().__init__("easy")


class MineSweeperMedium(MineSweeper):
    def __init__(self):
        super().__init__("medium")


class MineSweeperHard(MineSweeper):
    def __init__(self):
        super().__init__("hard")
