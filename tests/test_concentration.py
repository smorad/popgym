import numpy as np

from popgym.envs.concentration import Concentration
from tests.base_env_test import AbstractTest


class TestConcentration(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = Concentration()

    def test_repeatedly_flip_faceup(self):
        # Ensure we cannot match facedown cards
        # with previously-matched faceup cards
        obs, _ = self.env.reset()
        r_c = 0
        obs, r, te, tr, _ = self.env.step(0)
        r_c += r
        cards = [obs[0]]
        found = False
        i = 0
        while not (te or tr) and not found:
            i += 1
            obs, r, te, tr, _ = self.env.step(i)
            r_c += r
            found = obs[i] in cards
            cards.append(obs[i])
        j = cards.index(obs[i])
        if not (te or tr) and i % 2 != 1:
            obs, r, te, tr, _ = self.env.step(j)
            r_c += r
        while not (te or tr):
            obs, r, te, tr, _ = self.env.step(i)
            r_c += r
            i, j = j, i

        self.assertAlmostEqual(
            r_c,
            self.env.success_reward_scale
            + (self.env.episode_length - 2) * self.env.failure_reward_scale,
        )

    def test_perfect_game(self):
        obs, _ = self.env.reset()
        ranks = self.env.deck.ranks_idx[self.env.deck.idx]
        done = False
        running_reward = 0
        for i in range(13):
            idx = np.where(ranks == i)[0]
            for j in range(4):
                self.assertFalse(done)
                obs, reward, terminated, truncated, info = self.env.step(idx[j])
                self.assertLessEqual(len(self.env.deck["in_play"]), 2)
                running_reward += reward
                if j % 2 == 1:
                    self.assertEqual(reward, self.env.success_reward_scale)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(running_reward, 1.0)

    def test_two_decks(self):
        e = Concentration(num_decks=2)
        obs, _ = e.reset()
        ranks = e.deck.ranks_idx[e.deck.idx]
        terminated = truncated = False
        running_reward = 0
        for i in range(13):
            idx = np.where(ranks == i)[0]
            for j in range(8):
                obs, reward, terminated, truncated, info = e.step(idx[j])
                running_reward += reward
                if j % 2 == 1:
                    self.assertEqual(reward, e.success_reward_scale)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(running_reward, 1.0)

    def test_perfect_game_colors(self):
        e = Concentration(deck_type="colors")
        obs, _ = e.reset()
        colors = e.deck.colors_idx[e.deck.idx]
        terminated = truncated = False
        running_reward = 0
        for i in range(2):
            idx = np.where(colors == i)[0]
            for j in range(26):
                obs, reward, terminated, truncated, info = e.step(idx[j])
                running_reward += reward
                if j % 2 == 1:
                    print(e.deck["in_play"])
                    self.assertEqual(reward, e.success_reward_scale)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(running_reward, 1.0)

    def test_worst_game(self):
        obs, _ = self.env.reset()
        terminated = truncated = False
        running_reward = 0
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = self.env.step(0)
            running_reward += reward

        self.assertAlmostEqual(running_reward, -1.0)
        self.assertTrue(truncated)
        self.assertFalse(terminated)

    def test_worst_game_colors(self):
        e = Concentration(deck_type="colors")
        obs, _ = e.reset()
        terminated = truncated = False
        running_reward = 0
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = e.step(0)
            running_reward += reward

        self.assertAlmostEqual(running_reward, -1.0)
        self.assertTrue(truncated)
        self.assertFalse(terminated)
