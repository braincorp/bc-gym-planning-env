""" Experiment, running an example planning environment based on real
data case of corridor with three boxes, along with some randomization """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import time

from bc_gym_planning_env.envs.base.env import PlanEnv
from bc_gym_planning_env.envs.rw_corridors.tdwa_test_environments import \
    get_random_maps_squeeze_between_obstacle_in_corridor_on_path
from bc_gym_planning_env.envs.base.params import EnvParams

if __name__ == '__main__':

    map_index = 4
    _, path, test_maps = get_random_maps_squeeze_between_obstacle_in_corridor_on_path()

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
        costmap=test_maps[map_index],
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
        time.sleep(0.1)
