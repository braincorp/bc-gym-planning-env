""" Running an example planning environment with custom-hand-built costmap and trajectory. """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import numpy as np
import time

from bc_gym_planning_env.envs.base.env import PlanEnv
from bc_gym_planning_env.envs.base.params import EnvParams
from bc_gym_planning_env.envs.base.maps import example_config, generate_trajectory_and_map_from_config


if __name__ == '__main__':
    map_config = example_config()
    path, costmap = generate_trajectory_and_map_from_config(map_config)

    env_params = EnvParams(
        iteration_timeout=1200,
        pose_delay=1,
        control_delay=0,
        state_delay=1,
        goal_spat_dist=1.0,
        goal_ang_dist=np.pi/2,
        dt=0.05,  # 20 Hz
        path_limiter_max_dist=5.0,
    )

    env = PlanEnv(
        costmap=costmap,
        path=path,
        params=env_params
    )

    env.reset()
    env.render()

    t = time.time()

    done = False

    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        # print "Passed {}".format(1/(time.time() - t))
        # t = time.time()
        time.sleep(0.1)
