import numpy as np

from popgym.envs.minesweeper import HiddenSquare, MineSweeper
from tests.base_env_test import AbstractTest


class TestMineSweeper(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = MineSweeper()

    def test_perfect(self):
        self.env.reset()
        xs, ys = np.where(self.env.hidden_grid == HiddenSquare.CLEAR)
        cum_rew = 0
        terminated = truncated = False
        for x, y in zip(xs, ys):
            self.assertFalse(terminated or truncated)
            obs, reward, terminated, truncated, info = self.env.step(np.array([x, y]))
            self.assertAlmostEqual(reward, self.env.success_reward_scale)
            cum_rew += reward

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(cum_rew, 1.0)

    def test_worst(self):
        self.env.reset()
        ts = self.env.max_episode_length - 1
        xs, ys = np.where(self.env.hidden_grid == HiddenSquare.CLEAR)
        cum_rew = 0
        terminated = truncated = False
        for i in range(ts):
            action = np.array([xs[0], ys[0]])
            obs, reward, terminated, truncated, info = self.env.step(action)
            cum_rew += reward

        self.assertFalse(terminated or truncated)

        xs, ys = np.where(self.env.hidden_grid == HiddenSquare.MINE)
        action = np.array([xs[0], ys[0]])
        obs, reward, terminated, truncated, info = self.env.step(action)
        cum_rew += reward
        self.assertTrue(terminated)

        self.assertAlmostEqual(cum_rew, -1.0)
