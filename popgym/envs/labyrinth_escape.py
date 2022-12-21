from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from popgym.core.maze import Actions, Cell, MazeEnv


class LabyrinthEscape(MazeEnv):
    """A maze environment where the agent receives negative rewards
     until it finds the goal.

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
        super().step(action)
        reward = self.neg_reward_scale
        terminated = False
        y, x = self.position
        if self.maze.grid[y, x] == Cell.GOAL:
            reward += 1.0
            terminated = True
        truncated = self.curr_step == self.max_episode_length - 1

        obs = self.get_obs(action)
        info = {"position": (x, y)}

        return obs, reward, terminated, truncated, info

    def render(self):
        print(self.tostring(start=True, end=True, agent=True, visited=True))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[gym.core.ObsType, Dict[str, Any]]:
        super().reset(seed=seed)
        # Based on free space
        x, y = self.position
        self.maze.grid[self.maze.end] = Cell.GOAL
        obs = self.get_obs(Actions.NONE)

        return obs, {"maze": str(self.maze.grid)}


class LabyrinthEscapeEasy(LabyrinthEscape):
    def __init__(self):
        super().__init__(maze_dims=(10, 10), episode_length=1024)


class LabyrinthEscapeMedium(LabyrinthEscape):
    def __init__(self):
        super().__init__(maze_dims=(14, 14), episode_length=1024)


class LabyrinthEscapeHard(LabyrinthEscape):
    def __init__(self):
        super().__init__(maze_dims=(18, 18), episode_length=1024)
