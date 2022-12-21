from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType

from popgym.core.maze import Actions, Cell, Explored, MazeEnv


class LabyrinthExplore(MazeEnv):
    """A maze environment where the agent receives reward for visiting
    new grid squares

    Args:
        maze_dims: (heigh, width) of the generated mazes in blocks
        episode_length: maximum length of an episode

    Returns:
        A gym environment

    """

    def __init__(self, maze_dims=(10, 10), episode_length=1024):
        super().__init__(maze_dims, episode_length)
        self.neg_reward_scale = -1 / self.max_episode_length

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        new_square = self.explored[tuple(self.move(action))] == Explored.NO
        super().step(action)
        reward = self.neg_reward_scale
        terminated = truncated = False
        y, x = self.position
        if new_square:
            reward = self.pos_reward_scale
        if self.curr_step == self.max_episode_length - 1:
            truncated = True
        free_mask = self.maze.grid != Cell.OBSTACLE
        visit_mask = self.explored == Explored.YES
        if np.all(free_mask == visit_mask):
            # Explored as much as possible
            terminated = True

        obs = self.get_obs(action)
        info = {"position": (x, y)}

        return obs, reward, terminated, truncated, info

    def render(self):
        print(self.tostring(start=True, end=False, agent=True, visited=True))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[gym.core.ObsType, Dict[str, Any]]:
        super().reset(seed=seed)
        # Based on free space
        self.pos_reward_scale = 1 / ((self.maze.grid == Cell.FREE).sum())
        y, x = self.position
        obs = self.get_obs(Actions.NONE)
        return obs, {"maze": str(self.maze.grid)}


class LabyrinthExploreEasy(LabyrinthExplore):
    def __init__(self):
        super().__init__(maze_dims=(10, 10), episode_length=1024)


class LabyrinthExploreMedium(LabyrinthExplore):
    def __init__(self):
        super().__init__(maze_dims=(14, 14), episode_length=1024)


class LabyrinthExploreHard(LabyrinthExplore):
    def __init__(self):
        super().__init__(maze_dims=(18, 18), episode_length=1024)
