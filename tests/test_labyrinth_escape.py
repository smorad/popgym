import unittest

from popgym.envs.labyrinth_escape import LabyrinthEscape


class TestLabyrinthEscape(unittest.TestCase):
    def test_step(self):
        e = LabyrinthEscape()
        e.reset()
        for i in range(1000):
            _, _, terminated, truncated, _ = e.step(e.action_space.sample())
            if terminated or truncated:
                e.reset()

    def test_tostring(self):
        e = LabyrinthEscape((6, 6))
        e.reset()
        e.tostring()

    def test_goal(self):
        e = LabyrinthEscape((6, 6))
        terminated = truncated = False
        e.reset()
        for i in range(5):
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = e.step(e.action_space.sample())
            self.assertTrue(e.curr_step < e.max_episode_length)
            e.reset()

    def test_known(self):
        e = LabyrinthEscape((8, 8))
        e.reset(seed=1)
        left, right, _, down = 0, 1, 2, 3
        actions = [down] * 5 + [left] * 2 + [down] * 2 + [right] * 6 + [down] * 1
        done = False
        cum_rew = 0
        for action in actions:
            self.assertFalse(done)
            obs, reward, terminated, truncated, info = e.step(action)
            cum_rew += reward

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        expected = 1.0 + len(actions) * e.neg_reward_scale
        self.assertAlmostEqual(cum_rew, expected)
