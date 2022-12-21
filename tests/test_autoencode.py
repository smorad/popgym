import numpy as np

from popgym.envs.autoencode import Autoencode, Mode
from tests.base_env_test import AbstractTest


class TestAutoencode(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = Autoencode()

    def test_full(self):
        obs, _ = self.env.reset()
        terminated = truncated = False
        seq = [obs[1]]
        reward_sum = 0
        while obs[0] == Mode.WATCH:
            obs, reward, terminated, truncated, _ = self.env.step(0)
            seq.append(np.array(obs[1]))
            reward_sum += reward
        self.assertFalse(terminated or truncated)
        self.assertEqual(reward_sum, 0)
        for i in range(len(seq)):
            self.assertFalse(terminated or truncated)
            a = seq.pop(-1)
            _, reward, terminated, truncated, _ = self.env.step(a)
            reward_sum += reward
        self.assertTrue(terminated or truncated)
        self.assertEqual(len(self.env.deck["system"]), 0)

        self.assertAlmostEqual(reward_sum, 1.0)
