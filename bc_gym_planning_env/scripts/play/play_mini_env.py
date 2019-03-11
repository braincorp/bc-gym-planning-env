""" Play the planning env as if it was a computer game"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from bc_gym_planning_env.envs.mini_env import RandomMiniEnv
from bc_gym_planning_env.utilities.gui import KeyCapturePlay


if __name__ == '__main__':
    for seed in range(1000):
        print(seed)

        env = RandomMiniEnv()
        env.seed(seed)

        play = KeyCapturePlay(env)

        play.pre_main_loop()
        while not play.done():
            play.before_env_step()
            play.env_step()
            play.post_env_step()
