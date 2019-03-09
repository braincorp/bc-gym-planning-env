"""
Simple playing for running random turn environment.
Steer with WASD and see what happens.
 """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


from bc_gym_planning_env.envs.synth_turn_env import RandomAisleTurnEnv
from bc_gym_planning_env.utilities.gui import KeyCapturePlay


if __name__ == '__main__':
    env = RandomAisleTurnEnv()
    env.seed(1337)
    play = KeyCapturePlay(env)

    play.pre_main_loop()
    while not play.done():
        play.before_env_step()
        play.env_step()
        play.post_env_step()
