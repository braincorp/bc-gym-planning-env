""" A synthetic robotic planning environment, where the goal is to make one aisle turn. """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from collections import OrderedDict

import attr
import numpy as np

from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.envs.base.params import EnvParams
from bc_gym_planning_env.envs.base.env import PlanEnv
from bc_gym_planning_env.envs.base import spaces
from bc_gym_planning_env.envs.base.maps import Wall
from bc_gym_planning_env.utilities.coordinate_transformations import from_global_to_egocentric, world_to_pixel
from bc_gym_planning_env.utilities.costmap_utils import extract_egocentric_costmap


@attr.s
class TurnParams(object):
    """ Parametrization of a specific turn """
    main_corridor_length = attr.ib(default=8, type=float)
    turn_corridor_length = attr.ib(default=5, type=float)
    turn_corridor_angle = attr.ib(default= 2 * np.pi / 8, type=float)   # angle of the turn
    main_corridor_width = attr.ib(default=1.0, type=float)                  # width of the main corridor
    turn_corridor_width = attr.ib(default=1.0, type=float)                  # width of the turn corridor
    margin = attr.ib(default=1.0, type=float)             # how much wider should we make the world?
    flip_arnd_oy = attr.ib(default=False, type=bool)   # should flip around oy axis?
    flip_arnd_ox = attr.ib(default=False, type=bool)   # should flip around ox axis?
    rot_theta = attr.ib(default=0, type=float)         # should rotate the whole problem?


@attr.s
class AisleTurnEnvParams(object):
    """ Parametrization of aisle-turning environment """
    env_params = attr.ib(factory=EnvParams)
    turn_params = attr.ib(factory=TurnParams)


def _draw_pts_in_standard_coords(d, h, alpha, z, w):
    """ Set the point coordinates in the vanilla setup: no rotations,
    no mirror images.
     I         J
     |         |
     |         G__l2___H
     |
   2h|     O  -L- - - -F
     |       alpha
     |     |
     |     K   D__l1___E
     |     |   |       .
     A          C       .
     x=-d  B   x=d     x = w cos


    :param d: width of the base corridor
    :param h: 2h is length of the base corridor
    :param alpha: alpha is the turn angle
    :param z: width of the corridor we will turn into
    :param w: half of the length of the corridor we will turn into.
    :return Tuple[np.ndarray]: geometric points as shown above
    """
    far_x = w

    ap = np.array([-d, -h])
    bp = np.array([0, -h])
    cp = np.array([d, -h])
    dp = np.array([d, d * np.tan(alpha) - z / np.cos(alpha)])
    ep = np.array([far_x, far_x * np.tan(alpha) - z / np.cos(alpha)])
    fp = np.array([far_x, far_x * np.tan(alpha)])
    gp = np.array([d, d * np.tan(alpha) + z / np.cos(alpha)])
    hp = np.array([far_x, far_x * np.tan(alpha) + z / np.cos(alpha)])
    ip = np.array([-d, h])
    jp = np.array([d, h])

    return ap, bp, cp, dp, ep, fp, gp, hp, ip, jp


def _generate_path_in_standard_coords(d, h, alpha, z, w):
    """
    Set the coarse path coordinates in the standard path

    :param d: width of the base corridor
    :param h: 2h is length of the base corridor
    :param alpha: alpha is the turn angle
    :param z: width of the corridor we will turn into
    :param w: half of the length of the corridor we will turn into.
    :return Tuple[np.ndarray]: Tuple of oriented way points == the path
    """
    rb = np.array([0, -h, np.pi/2])
    rk = np.array([0, d * np.tan(alpha) - z / np.cos(alpha), np.pi/2])
    rl = np.array([d, d * np.tan(alpha), alpha])
    rf = np.array([w * np.cos(alpha), w * np.cos(alpha) * np.tan(alpha), alpha])

    return rb, rk, rl, rf


def _rotation_matrix(theta):
    """
    Get a rotation matrix for operation of rotation of the linear space
    around the 0,0 point.
    :param theta float: rotation angle
    :return np.array: 2d rotation matrix
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def path_and_costmap_from_config(params):
    """
    Generate the actual path and  turn

    :param params TurnParams: information about the turn
    :return: Tuple[ndarray(N, 3), Costmap2D]
    """
    # we assume right turn, we can always flip it
    turn_params = params.turn_params

    hh = turn_params.main_corridor_length / 2
    w = turn_params.turn_corridor_length / 2
    alpha = turn_params.turn_corridor_angle
    dd = turn_params.main_corridor_width
    z = turn_params.turn_corridor_width
    margin = turn_params.margin
    flip_arnd_oy = turn_params.flip_arnd_oy
    flip_arnd_ox = turn_params.flip_arnd_ox
    rot_theta = turn_params.rot_theta

    pts = _draw_pts_in_standard_coords(dd, hh, alpha, z, w)
    oriented_way_pts = _generate_path_in_standard_coords(dd, hh, alpha, z, w)

    # Maybe transform the points
    rot_mtx = _rotation_matrix(rot_theta)

    flipping_mtx = np.array(
        [[-1. if flip_arnd_oy else 1., 0.],
         [0., -1. if flip_arnd_ox else 1.]],
    )
    transform_mtx = np.dot(rot_mtx, flipping_mtx)

    new_pts = []

    for pt in pts:
        new_pt = np.dot(transform_mtx, pt)
        new_pts.append(new_pt)

    new_oriented_way_pts = []
    for pt in oriented_way_pts:
        x, y, t = pt
        nx, ny = np.dot(transform_mtx, np.array([x, y]))
        new_angle = t
        if flip_arnd_ox:
            new_angle = -new_angle
        if flip_arnd_oy:
            new_angle = np.pi - new_angle
        new_angle = np.mod(new_angle + rot_theta, 2 * np.pi)
        new_pt = np.array([nx, ny, new_angle])
        new_oriented_way_pts.append(new_pt)

    a, _, c, d, e, _, g, h, i, j = new_pts   # pylint: disable=unbalanced-tuple-unpacking
    rb, rk, rl, rf = new_oriented_way_pts    # pylint: disable=unbalanced-tuple-unpacking
    all_pts = np.array(list(new_pts))

    min_x = all_pts[:, 0].min()
    max_x = all_pts[:, 0].max()
    min_y = all_pts[:, 1].min()
    max_y = all_pts[:, 1].max()

    world_size = abs(max_x - min_x) + 2 * margin, abs(max_y - min_y) + 2 * margin
    world_origin = min_x - margin, min_y - margin

    obstacles = [
        Wall(from_pt=a, to_pt=i),
        Wall(from_pt=c, to_pt=d),
        Wall(from_pt=d, to_pt=e),
        Wall(from_pt=j, to_pt=g),
        Wall(from_pt=g, to_pt=h)
    ]

    static_path = np.array([rb, rk, rl, rf])

    static_map = CostMap2D.create_empty(
        world_size=world_size,  # x width, y height
        resolution=params.env_params.resolution,
        world_origin=world_origin
    )

    for obs in obstacles:
        static_map = obs.render(static_map)

    return static_path, static_map


class AisleTurnEnv(PlanEnv):
    """
    Robotic Planning Environment in which the task
    is to turn right into an aisle based on provided
    config (of type AisleTurnEnvParams), that specifies
    the geometry of the turn.
    """
    def __init__(self, config):
        """
        Initialize turn environment.
        :param config TurnParams: parametrization of the specific env turn.
        """
        self._config = config
        path, costmap = path_and_costmap_from_config(config)
        super(AisleTurnEnv, self).__init__(costmap, path, config.env_params)

    def get_robot(self):
        """
        Get the robot.
        :return IRobot: robot that is part of this environment
        """
        return self._robot


class RandomAisleTurnEnv(object):
    """
    AisleTurnEnv where the turn geometry is drawn randomly
    over and over again.

    if draw_new_turn_on_reset is True, it samples new turn
    on env.reset(), otherwise it keeps showing same turn.
    """
    def __init__(self, params=None, draw_new_turn_on_reset=True, seed=None, rng=None):
        """ Initialize Random Aisle Turn Planning Environment
        :param params EnvParams: environment parameters that can be used to customize the benchmark.
                           These are parameters of the base PlanEnv, and they are passed down there.
        :param draw_new_turn_on_reset bool: should we draw a new turn on each reset, or just keep redoing the first one.
        :param seed int: random seed
        :param rng np.RandomState: random number generator
        """
        if rng is None:
            self._rng = np.random.RandomState(1001)
        else:
            self._rng = rng

        self.seed(seed)
        self._draw_new_turn_on_reset = draw_new_turn_on_reset

        turn_params = self._draw_random_turn_params()
        if params is None:
            params = EnvParams()
        self._env_params = params
        self.config = AisleTurnEnvParams(turn_params=turn_params, env_params=self._env_params)
        self._env = AisleTurnEnv(self.config)

        self.action_space = self._env.action_space
        # self.observation_space = self._env.observation_space

    def seed(self, seed=None):
        """
        Set the random generator with given seed, so you can control the sequence of
        environments generated.
        :param seed int: the seed you want to use
        """
        if seed is not None:
            self._rng.seed(seed)

    def step(self, action):
        """
        Run one timestep of the planning environment's dynamics, until end of
        episode is reached.

        Returns:
            observation (Observation): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls have no point
            info (dict): contains auxiliary diagnostic information (e.g. helpful for debugging)

        :param action: (wheel_v, wheel_angle)
        :return Tuple[Observation, float, bool, Dict]: the stuff env shuold return
        """
        return self._env.step(action)

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Resets the 'done' state as well.

        :return Observation:  observation on reset of the environment,
                              to be fed to agent as the initial observation.
        """
        if self._draw_new_turn_on_reset:
            turn_params = self._draw_random_turn_params()
            config = AisleTurnEnvParams(turn_params=turn_params, env_params=self._env_params)
            self._env = AisleTurnEnv(config)

        return self._env.reset()

    def render(self, mode='human'):
        """
        Render human-friendly representation of the environment on the screen.
        :param mode str: the mode of rendering, currently only 'human' works
        :return np.ndarray: the human - friendly depiction of the environment
        """
        return self._env.render(mode)

    def close(self):
        """ Do whatever you need to do on closing: release the resources etc. """
        self._env.close()

    def get_state(self):
        """ Get the state of the base environment underlying this simulation.
        :return State: the state of the underlying base environment
        """
        return self._env.get_state()

    def set_state(self, state):
        """ Set the state of the environment
        :param state State: the state of the base planning environment
        """
        self._env.set_state(state)

    def _draw_random_turn_params(self):
        """
        Draw random turn params

        :return TurnParams: Random turn params
        """
        return TurnParams(
            main_corridor_length=self._rng.uniform(10, 16),
            turn_corridor_length=self._rng.uniform(4, 12),
            turn_corridor_angle=self._rng.uniform(-3./8. * np.pi, 3./8.*np.pi),
            main_corridor_width=self._rng.uniform(0.5, 1.5),
            turn_corridor_width=self._rng.uniform(0.5, 1.5),
            flip_arnd_oy=bool(self._rng.rand() < 0.5),
            flip_arnd_ox=bool(self._rng.rand() < 0.5),
            rot_theta=self._rng.uniform(0, 2*np.pi)
        )


class ColoredCostmapRandomAisleTurnEnv(RandomAisleTurnEnv):
    """ Random Aisle Turn environment, but the observation is a colored costmap. """
    def __init__(self):
        super(ColoredCostmapRandomAisleTurnEnv, self).__init__()
        # TODO: Will need some trickery to do it fully openai gym style
        # As openai gym style requires knowing resolution of the image up front
        self.observation_space = spaces.Box(low=0, high=255, shape=(510, 264, 1), dtype=np.uint8)

    def step(self, action):
        """
        Run one timestep of the planning environment's dynamics, until end of
        episode is reached.

        Returns:
            observation (np.ndarray): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls have no point
            info (dict): contains auxiliary diagnostic information (e.g. helpful for debugging)

        :param action: (wheel_v, wheel_angle)
        :return Tuple[np.ndarray, float, bool, Dict]: the stuff env shuold return
        """
        rich_obs, reward, done, info = super(ColoredCostmapRandomAisleTurnEnv, self).step(action)
        obs = rich_obs.costmap.get_data()   # pylint: disable=no-member
        obs = np.expand_dims(obs, -1)
        return obs, reward, done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Resets the 'done' state as well.

        :return np.ndarray:  observation on reset of the environment,
                              to be fed to agent as the initial observation.
        """
        rich_obs = super(ColoredCostmapRandomAisleTurnEnv, self).reset()
        obs = rich_obs.costmap.get_data()   # pylint: disable=no-member
        obs = np.expand_dims(obs, -1)
        return obs


class ColoredEgoCostmapRandomAisleTurnEnv(RandomAisleTurnEnv):
    """ Random Aisle Turn environment, but the observation is a colored egocentric
    costmap and normalized goal coordinates. """

    def __init__(self):
        super(ColoredEgoCostmapRandomAisleTurnEnv, self).__init__()
        # TODO: Will need some trickery to do it fully openai gym style
        # As openai gym style requires knowing resolution of the image up front
        self._egomap_x_bounds = np.array([-0.5, 3.])  # aligned with robot's direction
        self._egomap_y_bounds = np.array([-2., 2.])  # orthogonal to robot's direction
        resulting_size = (self._egomap_x_bounds[1] - self._egomap_x_bounds[0],
                          self._egomap_y_bounds[1] - self._egomap_y_bounds[0])

        pixel_size = world_to_pixel(np.asarray(resulting_size, dtype=np.float64), np.zeros((2,)), resolution=0.03)
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
        robot_pose = self._env.get_robot().get_pose()

        ego_costmap = extract_egocentric_costmap(
            costmap,
            robot_pose,
            resulting_origin=(self._egomap_x_bounds[0], self._egomap_y_bounds[0]),
            resulting_size=(self._egomap_x_bounds[1] - self._egomap_x_bounds[0],
                            self._egomap_y_bounds[1] - self._egomap_y_bounds[0]))

        ego_path = from_global_to_egocentric(rich_observation.path, robot_pose)
        obs = np.expand_dims(ego_costmap.get_data(), -1)
        normalized_goal = ego_path[-1, :2] / ego_costmap.world_size()
        normalized_goal = normalized_goal / np.linalg.norm(normalized_goal)
        return OrderedDict((('environment', obs),
                            ('goal', np.expand_dims(normalized_goal, -1))))

    def step(self, action):
        """
        Run one timestep of the planning environment's dynamics, until end of
        episode is reached.

        Returns:
            observation (OrderedDict): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls have no point
            info (dict): contains auxiliary diagnostic information (e.g. helpful for debugging)

        :param action: (wheel_v, wheel_angle)
        :return Tuple[OrderedDict, float, bool, Dict]: the stuff env shuold return
        """
        """ Action is a motion command """
        rich_obs, reward, done, info = super(ColoredEgoCostmapRandomAisleTurnEnv, self).step(action)
        obs = self._extract_egocentric_observation(rich_obs)
        return obs, reward, done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Resets the 'done' state as well.

        :return OrderedDict:  observation on reset of the environment,
                              to be fed to agent as the initial observation.
        """
        rich_obs = super(ColoredEgoCostmapRandomAisleTurnEnv, self).reset()
        obs = self._extract_egocentric_observation(rich_obs)
        return obs
