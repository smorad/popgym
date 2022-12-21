import math

from popgym.envs.higher_lower import HigherLower
from tests.base_env_test import AbstractTest


class TestHigherLower(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = HigherLower()

    def test_episode_higher(self):
        terminated = truncated = False
        o, _ = self.env.reset()
        obs_list = [o]
        reward_list = []
        decklen = 52
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = self.env.step(0)
            obs_list.append(obs)
            reward_list.append(reward)
        pairs = list(zip(obs_list[:-1], obs_list[1:]))
        pred_correct = [j > i for i, j in pairs]
        pred_incorrect = [j < i for i, j in pairs]
        pred_rew = (sum(pred_correct) - sum(pred_incorrect)) / decklen
        actual_rew = sum(reward_list)

        self.assertTrue(math.isclose(pred_rew, actual_rew))
