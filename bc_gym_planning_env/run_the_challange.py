import itertools

import attr
import numpy as np

from bc_gym_planning_env.envs.base.action import Action
from bc_gym_planning_env.envs.base.env import PlanEnv
from bc_gym_planning_env.envs.base.obs import Observation
from bc_gym_planning_env.envs.base.params import EnvParams
from bc_gym_planning_env.envs.t_junction_env import TJunction
from bc_gym_planning_env.utilities.coordinate_transformations import normalize_angle


class RandomActor:
    rng: np.random.RandomState = attr.Factory(np.random.RandomState)

    def act(self, obs: Observation) -> Action:
        forward_velocity = np.random.uniform(0, 0.6)
        wanted_wheel_angle = np.random.uniform(-np.pi / 12.0, np.pi / 12.0)

        return Action(command=np.array([forward_velocity, wanted_wheel_angle]))


if __name__ == '__main__':
    geometry = TJunction()

    env_params = EnvParams()

    for start_position, end_position in itertools.permutations(['left', 'right', 'bottom'], 2):

        path = geometry.get_path(start_position, end_position)

        env = PlanEnv(
            costmap=geometry.get_costmap(),
            path=path,
            params=env_params
        )

        actor = RandomActor()

        done = False
        obs = env.reset()

        while not done:
            action = actor.act(obs)
            obs, reward, done, _ = env.step(action)
            env.render()

        spatial_diff = np.linalg.norm(obs.pose[:2] - path[-1, :2])
        angular_diff = np.abs(normalize_angle(obs.pose[2] - path[-1, 2]))

        assert spatial_diff < env_params.goal_spat_dist, f"Spatial diff is {spatial_diff} > {env_params.goal_spat_dist}"
        assert angular_diff < env_params.goal_ang_dist, f"Angular diff is {angular_diff} > {env_params.goal_ang_dist}"





