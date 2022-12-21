from popgym.envs.repeat_first import RepeatFirst
from tests.base_env_test import AbstractTest


class TestRepeatFirst(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = RepeatFirst()

    def test_perfect(self):
        terminated = truncated = False
        init_obs, _ = self.env.reset()
        is_start, init_item = init_obs
        self.assertEqual(is_start, 1)
        reward = 0
        for i in range(51):
            self.assertFalse(terminated or truncated)
            obs, rew, terminated, truncated, info = self.env.step(init_item)
            is_start, item = obs
            self.assertTrue(item < 4)
            self.assertEqual(is_start, 0)
            reward += rew

        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertAlmostEqual(1.0, reward)
