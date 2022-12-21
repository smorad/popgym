import pytest
from gymnasium.utils.env_checker import check_env

from popgym import ALL_ENVS


class TestEnvs:
    @pytest.mark.parametrize("env", ALL_ENVS)
    def test_no_warn(self, env):
        e = env()
        e.reset()
        check_env(e, skip_render_check=True)
