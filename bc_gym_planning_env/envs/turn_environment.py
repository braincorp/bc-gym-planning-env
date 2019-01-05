from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from collections import OrderedDict

import attr
import numpy as np
from gym.envs.registration import register

import gym.spaces as spaces
from gym import Env

from bc_gym_planning_env.utilities.coordinate_transformations import from_global_to_egocentric
from bc_gym_planning_env.utilities.costmap_2d_python import CostMap2D
from bc_gym_planning_env.envs.internals.plan_env import EnvParams, PlanEnv
from bc_gym_planning_env.envs.internals.maps import Wall
from bc_gym_planning_env.utilities.costmap_utils import extract_egocentric_costmap
from bc_gym_planning_env.utilities.path_tools import world_to_pixel


@attr.s
class TurnParams(object):
    main_corridor_length = attr.ib(default=8, type=float)
    turn_corridor_length = attr.ib(default=5, type=float)
    turn_corridor_angle = attr.ib(default= 2 * np.pi / 8, type=float)   # angle of the turn
    main_corridor_width = attr.ib(default=1.0, type=float)                  # width of the main corridor
    turn_corridor_width = attr.ib(default=1.0, type=float)                  # width of the turn corridor
    margin = attr.ib(default=1.0, type=float)             # how much wider should we make the world?


@attr.s
class AisleTurnEnvParams(object):
    env_params = attr.ib(factory=EnvParams)
    resolution = attr.ib(default=0.03, type=float)
    turn_params = attr.ib(factory=TurnParams)


def path_and_costmap_from_config(params):
    """
    Generate the actual path and  turn

    :param turn_params TurnParams: information about the turn
    :return: Tuple[ndarray(N, 3), Costmap2D]
    """
    # we assume right turn, we can always flip it
    # see playground.py for explaination of the point names
    turn_params = params.turn_params

    h = turn_params.main_corridor_length / 2
    w = turn_params.turn_corridor_length / 2
    alpha = turn_params.turn_corridor_angle
    d = turn_params.main_corridor_width
    z = turn_params.turn_corridor_width
    margin = turn_params.margin

    far_x = w

    O = 0, 0

    A = np.array([-d, -h])
    B = np.array([0, -h])
    C = np.array([d, -h])
    D = np.array([d, d * np.tan(alpha) - z / np.cos(alpha)])
    E = np.array([far_x, far_x * np.tan(alpha) - z / np.cos(alpha)])
    F = np.array([far_x, far_x * np.tan(alpha)])
    G = np.array([d, d * np.tan(alpha) + z / np.cos(alpha)])
    H = np.array([far_x, far_x * np.tan(alpha) + z / np.cos(alpha)])
    I = np.array([-d, h])
    J = np.array([d, h])

    rB = np.array([0, -h, np.pi/2])
    rK = np.array([0, d * np.tan(alpha) - z / np.cos(alpha), np.pi/2])
    rL = np.array([d, d * np.tan(alpha), alpha])
    rF = np.array([w * np.cos(alpha), w * np.cos(alpha) * np.tan(alpha), alpha])

    all_pts = np.array([A, B, C, D, E, F, G, H, I, J, O])

    min_x = all_pts[:, 0].min()
    max_x = all_pts[:, 0].max()
    min_y = all_pts[:, 1].min()
    max_y = all_pts[:, 1].max()

    world_size = abs(max_x - min_x) + 2 * margin, abs(max_y - min_y) + 2 * margin
    world_origin = min_x - margin, min_y - margin

    obstacles = [
        Wall(from_pt=A, to_pt=I),
        Wall(from_pt=C, to_pt=D),
        Wall(from_pt=D, to_pt=E),
        Wall(from_pt=J, to_pt=G),
        Wall(from_pt=G, to_pt=H)
    ]

    coarse_static_path = np.array([rB, rK, rL, rF])

    static_map = CostMap2D.create_empty(
        world_size=world_size,  # x width, y height
        resolution=params.resolution,
        world_origin=world_origin
    )

    for obs in obstacles:
        static_map = obs.render(static_map)

    return coarse_static_path, static_map


class AisleTurnEnv(PlanEnv):
    def __init__(self, config):
        self._config = config
        path, costmap = path_and_costmap_from_config(config)
        super(AisleTurnEnv, self).__init__(costmap, path, config.env_params)


class RandomAisleTurnEnv(Env):
    def __init__(self, draw_new_turn_on_reset=True, seed=None):
        self.seed(seed)
        self._draw_new_turn_on_reset = draw_new_turn_on_reset

        turn_params = self._draw_random_turn_params()
        config = AisleTurnEnvParams(turn_params=turn_params)
        self._env = AisleTurnEnv(config)

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        if self._draw_new_turn_on_reset:
            turn_params = self._draw_random_turn_params()
            config = AisleTurnEnvParams(turn_params=turn_params)
            self._env = AisleTurnEnv(config)

        return self._env.reset()

    def render(self, mode='human'):
        return self._env.render(mode)

    def close(self):
        return self._env.close()

    def _draw_random_turn_params(self):
        return TurnParams(
            main_corridor_length=np.random.uniform(10, 16),
            turn_corridor_length=np.random.uniform(4, 12),
            turn_corridor_angle=np.random.uniform(-3./8. * np.pi, 3./8.*np.pi),
            main_corridor_width=np.random.uniform(0.5, 1.5),
            turn_corridor_width=np.random.uniform(0.5, 1.5)
        )



class ColoredCostmapRandomAisleTurnEnv(RandomAisleTurnEnv):
    def __init__(self):
        super(ColoredCostmapRandomAisleTurnEnv, self).__init__()
        # TODO: Will need some trickery to do it fully openai gym style
        # As openai gym style requires knowing resolution of the image up front
        self.observation_space = spaces.Box(low=0, high=255, shape=(510, 264, 1), dtype=np.uint8)

    def step(self, motion_command):
        rich_obs, reward, done, info = super(ColoredCostmapRandomAisleTurnEnv, self).step(motion_command)
        obs = rich_obs.costmap.get_data()
        obs = np.expand_dims(obs, -1)
        return obs, reward, done, info

    def reset(self):
        rich_obs = super(ColoredCostmapRandomAisleTurnEnv, self).reset()
        obs = rich_obs.costmap.get_data()
        obs = np.expand_dims(obs, -1)
        return obs


class ColoredEgoCostmapRandomAisleTurnEnv(RandomAisleTurnEnv):
    def __init__(self):
        super(ColoredEgoCostmapRandomAisleTurnEnv, self).__init__()
        # TODO: Will need some trickery to do it fully openai gym style
        # As openai gym style requires knowing resolution of the image up front
        self._egomap_x_bounds = np.array([-0.5, 3.])  # aligned with robot's direction
        self._egomap_y_bounds = np.array([-2., 2.])  # orthogonal to robot's direction
        resulting_size = (self._egomap_x_bounds[1] - self._egomap_x_bounds[0],
                          self._egomap_y_bounds[1] - self._egomap_y_bounds[0])

        pixel_size = world_to_pixel(resulting_size, np.zeros((2,)), resolution=0.03)
        data_shape = (pixel_size[1], pixel_size[0], 1)
        self.observation_space = spaces.Dict(OrderedDict((
            ('environment', spaces.Box(low=0, high=255, shape=data_shape, dtype=np.uint8)),
            ('goal', spaces.Box(low=-1., high=1., shape=(2, 1), dtype=np.float64))
        )))

    def _extract_egocentric_observation(self, rich_observation):
        """Extract egocentric map and path from rich observation
        :param rich_observation RichObservation: observation, generated by a parent class
        :return (array(W, H)[uint8], array(N, 3)[float]): egocentric obstacle data and a path
        """
        costmap = rich_observation.costmap
        robot_pose = self._robot.get_pose()

        ego_costmap = extract_egocentric_costmap(
            costmap,
            robot_pose,
            resulting_origin=(self._egomap_x_bounds[0], self._egomap_y_bounds[0]),
            resulting_size=(self._egomap_x_bounds[1] - self._egomap_x_bounds[0],
                            self._egomap_y_bounds[1] - self._egomap_y_bounds[0]))

        ego_path = from_global_to_egocentric(rich_observation.path, robot_pose)
        obs = np.expand_dims(ego_costmap.get_data(), -1)
        normalized_goal = ego_path[-1, :2] / ego_costmap.world_size()
        normalized_goal = np.clip(normalized_goal, (-1., -1.), (1., 1.))
        return OrderedDict((('environment', obs),
                            ('goal', np.expand_dims(normalized_goal, -1))))

    def step(self, motion_command):
        rich_obs, reward, done, info = super(ColoredEgoCostmapRandomAisleTurnEnv, self).step(motion_command)
        obs = self._extract_egocentric_observation(rich_obs)
        return obs, reward, done, info

    def reset(self):
        rich_obs = super(ColoredEgoCostmapRandomAisleTurnEnv, self).reset()
        obs = self._extract_egocentric_observation(rich_obs)
        return obs


register(
    id='RandomTurnRoboPlanning-v0',
    entry_point='bc_gym_planning_env.envs.turn_environment:RandomAisleTurnEnv',
    kwargs=dict(seed=1)
)

register(
    id='CostmapAsImgRandomTurnRoboPlanning-v0',
    entry_point='bc_gym_planning_env.envs.turn_environment:ColoredCostmapRandomAisleTurnEnv'
)

register(
    id='EgoCostmapAsImgRandomTurnRoboPlanning-v0',
    entry_point='bc_gym_planning_env.envs.turn_environment:ColoredEgoCostmapRandomAisleTurnEnv'
)
