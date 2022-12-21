import unittest

import numpy as np

from popgym.envs.labyrinth_explore import LabyrinthExplore


class TestLabyrinthExplore(unittest.TestCase):
    def test_step(self):
        e = LabyrinthExplore()
        obs, _ = e.reset()
        self.assertEqual(obs.dtype, np.int32)
        terminated = truncated = False
        for i in range(1000):
            obs, _, terminated, truncated, _ = e.step(e.action_space.sample())
            self.assertEqual(obs.dtype, np.int32)
            if terminated or truncated:
                obs = e.reset()
                self.assertEqual(obs.dtype, np.int32)

    def test_noclip(self):
        e = LabyrinthExplore()
        terminated = truncated = False
        e.reset()
        for i in range(5):
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = e.step(e.action_space.sample())

            visit_mask = e.explored == 1
            obstacle_mask = e.maze.grid == 1
            self.assertFalse(np.any(visit_mask * obstacle_mask == 1))
            e.reset()

    def test_known(self):
        e = LabyrinthExplore((6, 6))
        e.reset(seed=0)
        left, right, up, down = 0, 1, 2, 3
        actions = (
            [left] * 5
            + [up] * 2
            + [right] * 2
            + [left] * 2
            + [down] * 2
            + [right] * 4
            + [up] * 4
            + [left] * 4
        )
        terminated = truncated = False
        cum_rew = 0
        for action in actions:
            self.assertFalse(terminated or truncated)
            obs, reward, terminated, truncated, info = e.step(action)
            cum_rew += reward

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        expected = 1.0 + 8 * e.neg_reward_scale
        self.assertAlmostEqual(cum_rew, expected)
