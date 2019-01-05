from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import gym

import bc_gym_planning_env


if __name__ == '__main__':
    env = gym.make('RandomTurnRoboPlanning-v0')

    import time
    while True:
        env.reset()
        env.render()

        done = False

        while not done:
            command = env.action_space.sample()
            command[0] += 0.3
            print("Applying: %s" % (command,))
            _, _, done, _ = env.step(command)
            env.render()
            time.sleep(0.05)
