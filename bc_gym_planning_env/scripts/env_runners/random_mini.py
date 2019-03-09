""" Run a random mini environment with egocentric costmap observation wrapper. """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
from shining_software.env_utils.envs.base.action import Action
from shining_software.env_utils.envs.mini_env import RandomMiniEnv
from shining_software.env_utils.envs.egocentric import EgocentricCostmap


if __name__ == '__main__':
    for seed in range(1000):
        print(seed)

        env = RandomMiniEnv()
        env = EgocentricCostmap(env)

        env.seed(seed)
        env.reset()
        env.render()

        done = False

        while not done:
            action = Action(command=np.array([0.3, 0.0]))
            _, _, done, _ = env.step(action)
            env.render()
