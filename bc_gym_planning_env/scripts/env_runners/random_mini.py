""" Run a random mini environment with egocentric costmap observation wrapper. """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
from bc_gym_planning_env.envs.base.action import Action
from bc_gym_planning_env.envs.mini_env import RandomMiniEnv
from bc_gym_planning_env.envs.egocentric import EgocentricCostmap


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
