import numpy as np

from popgym.envs.multiarmed_bandit import MultiarmedBandit
from tests.base_env_test import AbstractTest


class TestMultiarmedBandit(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = MultiarmedBandit()

    def test_best(self):
        obs, info = self.env.reset()
        bandits = info["bandits"]
        action = np.argmax(bandits)
        terminated = truncated = False
        reward = 0
        expected_value = (
            self.env.max_episode_length * np.max(bandits)
            - self.env.max_episode_length * (1 - np.max(bandits))
        ) / self.env.max_episode_length

        for i in range(self.env.max_episode_length):
            self.assertFalse(terminated or truncated)
            obs, rew, terminated, truncated, info = self.env.step(action)
            reward += rew

        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertTrue(np.abs(expected_value - reward) < 0.1)
