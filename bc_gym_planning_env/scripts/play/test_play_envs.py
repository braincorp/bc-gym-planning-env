from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from bc_gym_planning_env.utilities.gui import KeyCapturePlay
from bc_gym_planning_env.envs.base.env import PlanEnv
from bc_gym_planning_env.envs.rw_corridors.tdwa_test_environments import\
    get_random_maps_squeeze_between_obstacle_in_corridor_on_path
from bc_gym_planning_env.envs.synth_turn_env import RandomAisleTurnEnv
from bc_gym_planning_env.envs.mini_env import RandomMiniEnv
from bc_gym_planning_env.envs.base.params import EnvParams
try:
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock


def test_key_capture_play():
    env = RandomAisleTurnEnv()
    env.seed(1337)
    play = KeyCapturePlay(env)
    play._display = MagicMock(return_value=ord('w'))

    play.pre_main_loop()
    while not play.done():
        play.before_env_step()
        play.env_step()
        play.post_env_step()


def test_play_real_corridor_env():
    map_index = 4
    _, path, test_maps = get_random_maps_squeeze_between_obstacle_in_corridor_on_path()

    env = PlanEnv(
        costmap=test_maps[map_index],
        path=path,
        params=EnvParams()
    )

    play = KeyCapturePlay(env)
    play._display = MagicMock(return_value=ord('w'))

    play.pre_main_loop()
    while not play.done():
        play.before_env_step()
        play.env_step()
        play.post_env_step()


def test_play_mini_env():
    for seed in range(10):
        print(seed)

        env = RandomMiniEnv()
        env.seed(seed)

        play = KeyCapturePlay(env)
        play._display = MagicMock(return_value=ord('w'))

        play.pre_main_loop()
        while not play.done():
            play.before_env_step()
            play.env_step()
            play.post_env_step()


if __name__ == '__main__':
    test_play_real_corridor_env()
    test_key_capture_play()
    test_play_mini_env()
