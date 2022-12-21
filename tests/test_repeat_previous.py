from popgym.envs.repeat_previous import RepeatPrevious
from tests.base_env_test import AbstractTest


class TestRepeatPrevious(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = RepeatPrevious()

    def test_k(self):
        e = RepeatPrevious(k=2)
        obs0, _ = e.reset()
        obs1, rew1, te1, tr1, info1 = e.step(0)
        self.assertFalse(te1 or tr1)
        obs2, rew2, te2, tr2, info2 = e.step(obs0[1])
        self.assertFalse(te2 or tr2)
        obs3, rew3, te3, tr3, info3 = e.step(obs1[1])
        self.assertFalse(te3 or tr3)

    def test_full(self):
        e = RepeatPrevious(k=2)
        obs1, _ = e.reset()
        cum_rew = 0
        obs2, reward, terminated, truncated, info = e.step(0)
        cum_rew += reward

        all_obs = [obs1[1], obs2[1]]
        should_act = [obs1[0], obs2[0]]
        self.assertEqual(should_act[0], 0)
        self.assertEqual(should_act[1], 1)

        iters = 2
        while not (terminated or truncated):
            a = all_obs[-2]
            obs, reward, terminated, truncated, info = e.step(a)
            self.assertEqual(obs[0], 1)
            all_obs.append(obs[1])
            cum_rew += reward
            iters += 1

        self.assertAlmostEqual(cum_rew, 1.0)

        self.assertEqual(iters, 52)

    def test_full_16(self):
        e = RepeatPrevious(k=16)
        o, _ = e.reset()
        all_obs = [o[1]]
        iters = 1
        cum_rew = 0
        while iters < 16:
            obs, reward, terminated, truncated, info = e.step(0)
            self.assertFalse(terminated or truncated)
            self.assertEqual(reward, 0)
            cum_rew += reward
            all_obs.append(obs[1])
            iters += 1

        while iters < 52:
            a = all_obs[-16]
            self.assertFalse(terminated or truncated)
            obs, reward, terminated, truncated, info = e.step(a)
            cum_rew += reward
            all_obs.append(obs[1])
            iters += 1

        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertAlmostEqual(cum_rew, 1.0)
