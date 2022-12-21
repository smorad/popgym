import random
from enum import IntEnum
from typing import Optional
from warnings import warn

import gymnasium as gym
import numpy as np
from mazelib import Maze
from mazelib.generate.HuntAndKill import HuntAndKill

from popgym.core.env import POPGymEnv


class Explored(IntEnum):
    NO = 0
    YES = 1


class Cell(IntEnum):
    FREE = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    HIDDEN = 4


class Actions(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    NONE = 4


class MazeEnv(POPGymEnv):
    """A base class for maze-based environments.

    Args:
        maze_dims: The width and height of the maze in blocks
        episode_length: The maximum length of an episode

    Returns:
        A gym environment
    """

    obs_requires_prev_action = True

    def __init__(self, maze_dims=(10, 10), episode_length=1024):
        assert maze_dims[0] % 2 == 0 and maze_dims[1] % 2 == 0, "Maze dims must be even"
        self.maze_dims = [m // 2 for m in maze_dims]
        self.max_episode_length = episode_length
        # This could be MultiBinary but that uses int8 dtype
        # which causes RLlib obs preprocessor to crash
        self.observation_space = gym.spaces.MultiDiscrete(
            np.array(9 * [len(Cell)] + [len(Actions)], dtype=np.int32)
            # np.full(9, len(Cell), dtype=np.int32)
        )
        self.action_space = gym.spaces.Discrete(len(Actions) - 1)
        self.state_space = gym.spaces.MultiDiscrete(
            np.full([m + 1 for m in maze_dims], len(Cell))
        )
        warn(
            "Maze environments have been shown to be solvable without the use "
            "of memory (e.g. MLP outperforms LSTM). We suggest not using them "
            "as POMDP benchmarks.",
            DeprecationWarning,
        )

    def get_obs(self, action):
        view = self.local_view(self.maze.grid).reshape(9).astype(np.int32)
        obs = np.concatenate([view, np.array([action])], dtype=np.int32)
        return obs

    def move(self, action):
        y, x = self.position
        act_map = {
            0: (y, x - 1),  # left
            1: (y, x + 1),  # right
            2: (y - 1, x),  # up
            3: (y + 1, x),  # down
        }
        new_pos = act_map[action]
        new_y = min(max(new_pos[0], 0), 2 * self.maze_dims[0])
        new_x = min(max(new_pos[1], 0), 2 * self.maze_dims[1])
        new_pos = (new_y, new_x)

        if self.maze.grid[new_pos] == 1:
            # can't go thru obstacle
            return self.position

        return new_pos

    def local_view(self, grid):
        y, x = self.position
        x_start = max(x - 1, 0)
        x_end = min(x + 1, self.maze.grid.shape[0])
        # Y goes from top to bottom
        y_start = max(y - 1, 0)
        y_end = min(y + 1, self.maze.grid.shape[1])

        view = grid[y_start : y_end + 1, x_start : x_end + 1]
        y_pad, x_pad = 3 - view.shape[0], 3 - view.shape[1]
        padded = np.pad(view, [(0, y_pad), (0, x_pad)], constant_values=1)
        return padded

    def tostring(self, start=False, end=False, agent=False, visited=False):
        """Return a string representation of the maze.
        This can also display the maze entrances/solutions IF they already exist.
        Args:
            entrances (bool): Do you want to show the entrances of the maze?
            solutions (bool): Do you want to show the solution to the maze?
        Returns:
            str: string representation of the maze
        """
        # build the walls of the grid
        txt = []
        for i, row in enumerate(self.maze.grid):
            col = []
            for j, cell in enumerate(row):
                if visited and self.explored[i, j] == 1:
                    col += ["++"]
                elif cell == 0:
                    col += ["  "]
                elif cell == 1:
                    col += ["██"]
            txt.append("".join(col))

        # insert the start and end points
        if start and self.maze.start:
            if self.maze.start != self.position:
                r, c = self.maze.start
                txt[r] = txt[r][: 2 * c] + "🟢" + txt[r][2 * (c + 1) :]
        if end and self.maze.end:
            if self.maze.end != self.position:
                r, c = self.maze.end
                txt[r] = txt[r][: 2 * c] + "🛑" + txt[r][2 * (c + 1) :]
        if agent:
            r, c = self.position
            txt[r] = txt[r][: 2 * c] + "🚎" + txt[r][2 * (c + 1) :]

        return "\n".join(txt)

    def step(self, action):
        self.position = self.move(action)
        y, x = self.position
        self.explored[y, x] = Explored.YES
        self.curr_step += 1

    def get_state(self) -> gym.core.ObsType:
        state = self.maze.grid.copy()
        state[self.explored == Explored.NO] = Cell.HIDDEN
        return state

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.maze = Maze()
        self.maze.generator = HuntAndKill(*self.maze_dims)
        self.maze.generate()
        self.maze.generate_entrances(start_outer=True)
        self.position = self.maze.start
        y, x = self.position
        self.explored = np.full_like(self.maze.grid, Explored.NO)
        self.maze.grid[y, x] = Cell.START
        self.explored[y, x] = Explored.YES
        self.curr_step = 0
