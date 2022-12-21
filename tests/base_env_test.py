import math
import random
import unittest

import numpy as np


def repeat(times):
    # https://stackoverflow.com/a/13606054
    def repeat_helper(f):
        def call_helper(*args):
            for i in range(0, times):
                f(*args)

        return call_helper

    return repeat_helper


def is_close(x, y):
    assert type(x) is type(y)
    if isinstance(x, (list, tuple)):
        close = all(map(is_close, x, y))
    elif isinstance(x, dict):
        close = all([is_close(x[key], y[key]) for key in x.keys()])
    elif isinstance(x, np.ndarray):
        close = np.allclose(x, y)
    elif isinstance(x, float):
        close = math.isclose(x, y)
    else:
        close = x == y
    return close


class AbstractTest(object):
    class POPGymTest(unittest.TestCase):
        def test_init(self):
            self.env.reset()

        def test_get_state(self):
            self.env.reset()
            self.env.get_state()

        def test_step(self):
            self.env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                obs, reward, terminated, truncated, info = self.env.step(
                    self.env.action_space.sample()
                )

        def test_spaces(self):
            obs, _ = self.env.reset()
            state = self.env.get_state()
            self.assertTrue(self.env.observation_space.contains(obs))
            self.assertTrue(self.env.state_space.contains(state))
            terminated = truncated = False
            while not (terminated or truncated):
                obs, reward, terminated, truncated, info = self.env.step(
                    self.env.action_space.sample()
                )
                state = self.env.get_state()
                self.assertTrue(self.env.observation_space.contains(obs))
                self.assertTrue(self.env.state_space.contains(state))

        # @repeat(5)
        def test_np_random(self):
            seed = random.randint(0, 1000)
            state_list = []
            obs_list = []
            reward_list = []
            terminated = truncated = False
            obs, _ = self.env.reset(seed=seed)
            obs_list.append(obs)
            state_list.append(self.env.get_state())
            action_list = []
            while not (terminated or truncated):
                action = self.env.action_space.sample()
                action_list.append(action)
                obs, rew, terminated, truncated, info = self.env.step(action)
                obs_list.append(obs)
                reward_list.append(rew)
                state_list.append(self.env.get_state())

            obs, _ = self.env.reset(seed=seed)
            state = self.env.get_state()
            i = 0
            self.assertTrue(is_close(obs, obs_list[i]))
            self.assertTrue(is_close(state, state_list[i]))
            terminated = truncated = False
            while not (terminated or truncated):
                obs, rew, terminated, truncated, info = self.env.step(action_list[i])
                state = self.env.get_state()
                self.assertTrue(is_close(obs, obs_list[i + 1]))
                self.assertTrue(is_close(state, state_list[i + 1]))
                self.assertEqual(rew, reward_list[i])
                self.assertEqual(terminated or truncated, i + 1 == len(reward_list))
                i += 1
