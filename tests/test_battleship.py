import numpy as np

from popgym.envs.battleship import Battleship
from tests.base_env_test import AbstractTest


class TestBattleship(AbstractTest.POPGymTest):
    def setUp(self) -> None:
        self.env = Battleship()

    def test_hit_miss(self):
        self.env.reset()
        ship_positions = np.where(self.env.board)  # (x_pos_arr, y_pos_arr)
        hit_pos = (ship_positions[0][0], ship_positions[1][0])
        no_ship_positions = np.where(~self.env.board.astype(bool))
        miss_pos = (no_ship_positions[0][0], no_ship_positions[1][0])
        obs1, reward1, terminated1, truncated1, info1 = self.env.step(
            hit_pos
        )  # test hit
        assert obs1 == 1
        assert reward1 == 1.0 / sum(self.env.ship_sizes)
        obs2, reward2, terminated2, truncated2, info2 = self.env.step(
            hit_pos
        )  # test duplicate hit position (should be a miss now)
        assert obs2 == 0
        assert reward2 == -1.0 / (
            self.env.max_episode_length - sum(self.env.ship_sizes)
        )
        obs3, reward3, terminated3, truncated3, info3 = self.env.step(
            miss_pos
        )  # test miss
        assert obs3 == 0
        assert reward3 == -1.0 / (
            self.env.max_episode_length - sum(self.env.ship_sizes)
        )

    def test_num_ships(self):
        self.env.reset()
        assert sum(self.env.ship_sizes) == np.sum(self.env.board)

    def test_reward_sum(self):
        self.env.reset()

        # Episode reward 1.0 (only guesses hits)
        reward = 0.0
        ship_positions = np.where(self.env.board)  # (x_pos_arr, y_pos_arr)
        for i in range(len(ship_positions[0])):
            hit_pos = (ship_positions[0][i], ship_positions[1][i])
            obsi, rewardi, terminatedi, truncatedi, infoi = self.env.step(hit_pos)
            reward += rewardi
            if i < len(ship_positions[0]) - 1:
                assert not (terminatedi or truncatedi)
            elif i == len(ship_positions[0]) - 1:
                assert terminatedi
        self.assertAlmostEqual(reward, 1.0)

        # Episode reward minimum
        self.env.reset()
        reward = 0.0
        no_ship_positions = np.where(~self.env.board.astype(bool))
        for i in range(self.env.max_episode_length):
            miss_pos = (
                no_ship_positions[0][i % len(no_ship_positions)],
                no_ship_positions[1][i % len(no_ship_positions)],
            )
            obsi, rewardi, terminatedi, truncatedi, infoi = self.env.step(miss_pos)
            reward += rewardi
            if i < self.env.max_episode_length - 1:
                assert not (terminatedi or truncatedi)
            elif i == self.env.max_episode_length - 1:
                assert truncatedi
        self.assertAlmostEqual(
            reward, -1.0 * self.env.max_episode_length / len(no_ship_positions[0])
        )

        # Episode reward 0.0
        self.env.reset()
        reward = 0.0
        for x in range(self.env.board_size):
            for y in range(self.env.board_size):
                obsi, rewardi, terminatedi, truncatedi, infoi = self.env.step((x, y))
                reward += rewardi
        self.assertAlmostEqual(reward, 0.0)
