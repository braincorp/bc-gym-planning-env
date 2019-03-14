from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from bc_gym_planning_env.utilities.gui import KeyCapturePlay
from bc_gym_planning_env.envs.synth_turn_env import RandomAisleTurnEnv
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


if __name__ == '__main__':
    test_key_capture_play()
