"""
A minimalistic local planning environment, in which there is only one "corner"
obstacle.
"""
from __future__ import division
from __future__ import absolute_import

import attr
import numpy as np
from bc_gym_planning_env.robot_models.tricycle_model import TricycleRobot

from bc_gym_planning_env.robot_models.robot_dimensions_examples import get_dimensions_example
from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.envs.base.maps import Wall
from bc_gym_planning_env.envs.base.env import PlanEnv, pose_collides
from bc_gym_planning_env.envs.base.params import EnvParams
from bc_gym_planning_env.utilities.coordinate_transformations import angle_diff, cart2pol, pol2cart, \
    normalize_angle
from bc_gym_planning_env.utilities.path_tools import pose_distances


class SpaceSeemsEmptyError(Exception):
    """ Exception representing that a particular space from which we are
     sampling is probably empty, as we have tried sampling quite a lot
     of points and they seem to be all out of constraints!
     """
    pass


@attr.s
class RandomMiniEnvParams(object):
    """ parametrizes a space from which we can draw MiniEnv specimens"""
    inner_h = attr.ib(default=3, type=float)
    inner_w = attr.ib(default=3, type=float)
    mid_margin = attr.ib(default=0.25, type=float)
    out_margin = attr.ib(default=1, type=float)

    min_obstacle_angle = attr.ib(default=np.pi/8., type=float)
    max_obstacle_angle = attr.ib(default=np.pi, type=float)

    lim_euc_dist = attr.ib(default=1000, type=float)
    lim_ang_dist = attr.ib(default=np.pi, type=float)

    angular_pose_noise_scale = attr.ib(default=np.pi / 2.0, type=float)

    env_params = attr.ib(factory=EnvParams)


@attr.s
class OrientedPoint(object):
    """ class representing an oriented point"""
    x = attr.ib(type=float)
    y = attr.ib(type=float)
    theta = attr.ib(type=float, converter=normalize_angle)

    def as_np(self):
        """
        Represent this point as numpy array.
        :return np.ndarray: representation of this unoriented point as numpy array.
        """
        return np.array([self.x, self.y, self.theta], dtype=float)


@attr.s
class Point(object):
    """ class representing an unoriented point"""
    x = attr.ib(type=float)
    y = attr.ib(type=float)

    def as_np(self):
        """
        Represent this unoriented point as numpy array.
        :return np.ndarray: representation of this unoriented point as numpy array.
        """
        return np.array([self.x, self.y], dtype=float)


@attr.s
class MiniEnvParams(object):
    """ parametrizes a space from which we can draw MiniEnv specimens"""
    h = attr.ib(type=float)
    w = attr.ib(type=float)

    start_pos = attr.ib(type=OrientedPoint)
    end_pos = attr.ib(type=OrientedPoint)

    obstacle_a = attr.ib(type=Point)
    obstacle_o = attr.ib(type=Point)
    obstacle_b = attr.ib(type=Point)

    env_params = attr.ib(factory=EnvParams)


def _point_from_polar(r, phi, tx, ty):
    """ make a Point from polar coordinates with some translation in cartesian coords.
    :param r float: radius part of the point's polar coordinates
    :param phi float: angle part of the point's polar coordinates
    :param tx float: additional translation to apply to this point
    :param ty flaot: additional translation to apply to this point
    :return Point: the translated point on the plane (in cartesian coordinates)
    """
    x, y = pol2cart(r, phi)
    return Point(x=x + tx, y=y + ty)


def _satisfies_constraints(pt, constraints):
    """ does given point satisfy all of the constraints?
    :param pt Point: point to test
    :param constraints List[Func]: a list of predicates,
                                      each taking a single pt
    :return bool: does this point satisfy all of the constraints
    """
    for f in constraints:
        if not f(pt):
            return False

    return True


def _sample_pose(rng, params, constraints):
    """ Try drawing an oriented point from the space where all
    specified constraints are satisfied.
    :param rng np.random.RandomState: independent random state
    :param params RandomMiniEnvParams: the params, as described above
    :param constraints List[Function(Point) -> bool]: Constraints that the points
                                                      have to satisfy
    :return Point: the sampled pose
    """
    for _ in range(1000):
        pt = OrientedPoint(
            x=rng.uniform(
                -params.inner_w / 2 - params.mid_margin,
                params.inner_w / 2 + params.mid_margin
            ),
            y=rng.uniform(
                -params.inner_h / 2 - params.mid_margin,
                params.inner_h / 2 + params.mid_margin
            ),
            theta=rng.uniform(0, 2*np.pi)
        )
        if _satisfies_constraints(pt, constraints):
            return pt

    raise ValueError("Something went wrong, the sampling space looks empty.")


def _sample_pose_circ(rng, params, constraints):
    """
    Sample the ending and starting pose from the circle,
    such that they are on the exactly opposite sides of the circle.
    :param rng np.random.RandomState: independent random state
    :param params RandomMiniEnvParams: the params, as described above
    :param constraints List[Function(Point) -> bool]: Constraints that the points
                                                      have to satisfy
    :return Tuple(OrientedPoint, OrientedPoint): starting pose and ending pose
    """
    for _ in range(1000):
        r = (params.inner_w + params.inner_h) / 4. + params.mid_margin
        r = min(r, params.lim_euc_dist)

        phi = rng.uniform(0, 2*np.pi)

        x, y = pol2cart(r, phi)
        theta = rng.uniform(0, 2*np.pi)
        # theta_e = rng.uniform(0, 2*np.pi)

        # while np.abs(theta - theta_e) >= params.lim_ang_dist:
        #     theta_e = rng.uniform(0, 2*np.pi)

        start_pt = OrientedPoint(x, y, theta)
        end_pt = OrientedPoint(-x, -y, theta)

        y_off = end_pt.y - start_pt.y
        x_off = end_pt.x - start_pt.x
        new_theta = np.arctan2(y_off, x_off)

        start_pt = OrientedPoint(x, y, new_theta)
        end_pt = OrientedPoint(-x, -y, new_theta)

        if _satisfies_constraints(start_pt, constraints) and \
                _satisfies_constraints(end_pt, constraints):
            return start_pt, end_pt

    raise SpaceSeemsEmptyError("Something went wrong, the sampling space looks empty.")


def _pick_pts_square_method(rng, params, obstacle_o, obstacle_start_angle, obstacle_angle):
    """
    Sample the ending and starting pose for the square method.
    Just sample uniformly pose from the inner square.
    The heading is aligned with the line that is spanned by these two points.
    :param rng np.random.RandomState: independent random state
    :param params RandomMiniEnvParams: the params, as described above
    :param obstacle_o Point: midpoint of the angle obstacle
    :param obstacle_start_angle float: starting angle of the obstacle
    :param obstacle_angle float: how wide angularly is the obstacle
    :return Tuple(OrientedPoint, OrientedPoint): starting pose and ending pose
    """
    # sample start pt
    def not_inside_obstacle(pt):
        _, phi = cart2pol(pt.x - obstacle_o.x, pt.y - obstacle_o.y)
        if (phi >= obstacle_start_angle) and (phi <= obstacle_start_angle + obstacle_angle):
            # inside obstacle
            return False
        elif (phi + 2 * np.pi >= obstacle_start_angle) and (phi + 2 * np.pi <= obstacle_start_angle + obstacle_angle):
            # inside obstacle
            return False
        else:
            return True

    start_pos = _sample_pose(rng, params, [not_inside_obstacle])

    # sample end pt
    def not_far_from_start_ang(pt):
        dist = angle_diff(start_pos.theta, pt.theta)
        return dist < params.lim_ang_dist

    def not_far_from_start_euc(pt):
        dist = np.linalg.norm(np.array([start_pos.x - pt.x, start_pos.y - pt.y]))
        return dist < params.lim_euc_dist

    end_pos = _sample_pose(rng, params,
                           [not_inside_obstacle,
                            not_far_from_start_ang,
                            not_far_from_start_euc])

    y = end_pos.y - start_pos.y
    x = end_pos.x - start_pos.x
    theta = np.arctan2(y, x)

    start_pos = OrientedPoint(start_pos.x, start_pos.y, theta)
    end_pos = OrientedPoint(end_pos.x, end_pos.y, theta)

    return start_pos, end_pos


def _pick_pts_circle_method(rng, params, obstacle_o, obstacle_start_angle, obstacle_angle):
    """
    Randomly sample a mini env. Actually you sample parameters of a mini env,
    but these can be used to construct a given MiniEnv.

    In this method we pick a circle of specified radius
    and we sample two points such that they are exactly opposite to each other.

    Of course we don't want the starting and ending pose to be inside the obstacle.

    :param rng np.random.RandomState: independent random state
    :param params RandomMiniEnvParams: the params, as described above
    :param obstacle_o Point: midpoint of the angle obstacle
    :param obstacle_start_angle float: starting angle of the obstacle
    :param obstacle_angle float: how wide angularly is the obstacle
    :return MiniEnvParams: sampled params of the environment
    """
    def not_inside_obstacle(pt):
        _, phi = cart2pol(pt.x - obstacle_o.x, pt.y - obstacle_o.y)
        if (phi >= obstacle_start_angle) and (phi <= obstacle_start_angle + obstacle_angle):
            # inside obstacle
            return False
        elif (phi + 2 * np.pi >= obstacle_start_angle) and (phi + 2 * np.pi <= obstacle_start_angle + obstacle_angle):
            # inside obstacle
            return False
        else:
            return True

    start_pos, end_pos = _sample_pose_circ(rng, params, [not_inside_obstacle])
    return start_pos, end_pos


def _sample_mini_env_params_no_final_check(params, rng):
    """
    Randomly sample parameters of the mini env. They can be used to
    initialize the mini env.

    In this method of sampling, there are three rectangles, contained in each other,
    with the same center. middle is inner with a margin, outer is middle with a margin.

    inner - obstacle corner point will be drawn from it
    middle - starting and finishing robot poses will be drawn from there
    outer - is synonymous with the whole effective environment

    :param params RandomMiniEnvParams: the params, as described above
    :param rng np.random.RandomState: independent random state
    :return MiniEnvParams: sampled params of the environment
    """
    obstacle_o = Point(
        x=rng.uniform(-params.inner_w / 2, params.inner_w / 2),
        y=rng.uniform(-params.inner_h / 2, params.inner_h / 2)
    )

    obstacle_start_angle = rng.uniform(0, 2*np.pi)
    obstacle_angle = rng.uniform(
        params.min_obstacle_angle,
        params.max_obstacle_angle
    )
    r = 3 * (params.inner_h + params.inner_w + params.mid_margin + params.out_margin)

    obstacle_a = _point_from_polar(r, obstacle_start_angle, obstacle_o.x, obstacle_o.y)
    obstacle_b = _point_from_polar(r, obstacle_start_angle + obstacle_angle, obstacle_o.x, obstacle_o.y)

    h = params.inner_h + 2 * params.mid_margin + 2 * params.out_margin
    w = params.inner_w + 2 * params.mid_margin + 2 * params.out_margin

    if rng.rand() < 0.7:
        start_pos, end_pos = _pick_pts_circle_method(rng, params, obstacle_o,
                                                     obstacle_start_angle, obstacle_angle)
    else:
        start_pos, end_pos = _pick_pts_square_method(rng, params, obstacle_o,
                                                     obstacle_start_angle, obstacle_angle)

    # inject some angular noise into start and end poses
    angular_noise = rng.uniform(- params.angular_pose_noise_scale / 2.0, params.angular_pose_noise_scale / 2.0)
    new_start_pos = OrientedPoint(start_pos.x, start_pos.y, start_pos.theta + angular_noise)
    angular_noise = rng.uniform(- params.angular_pose_noise_scale / 2.0, params.angular_pose_noise_scale / 2.0)
    new_end_pos = OrientedPoint(end_pos.x, end_pos.y, end_pos.theta + angular_noise)

    out_params = MiniEnvParams(
        h, w, new_start_pos, new_end_pos, obstacle_a, obstacle_o, obstacle_b, params.env_params
    )

    return out_params


def _sample_mini_env_params(gen_params, rng):
    """
    Sample mini env instance based on passed params and check if the start and end poses
    are not colliding.
    If you can't sample, raise an error.

    :param gen_params RandomMiniEnvParams: params of the random mini env
    :param rng np.random.RandomState: independent random state
    :return MiniEnvParams: sampled params of the environment
    """
    for _ in range(1000):
        # We need to check for collisions of start and
        try:
            drawn_params = _sample_mini_env_params_no_final_check(gen_params, rng)
        except SpaceSeemsEmptyError:
            continue

        costmap, static_path = prepare_map_and_path(drawn_params)
        robot = TricycleRobot(dimensions=get_dimensions_example(gen_params.env_params.robot_name))
        robot.set_front_wheel_angle(gen_params.env_params.initial_wheel_angle)

        x, y, angle = static_path[0]
        first_pose_collides = pose_collides(x, y, angle, robot, costmap)

        x, y, angle = static_path[1]
        second_pose_collides = pose_collides(x, y, angle, robot, costmap)

        collides = first_pose_collides or second_pose_collides

        # are beg and goal poses not immediately too close
        cart_dist, ang_dist = pose_distances(static_path[0], static_path[1])
        cart_near = cart_dist < gen_params.env_params.goal_spat_dist
        ang_near = ang_dist < gen_params.env_params.goal_ang_dist
        too_close = cart_near and ang_near

        if not collides and not too_close:
            return drawn_params

    raise ValueError("Something went wrong, the sampling space looks empty.")


def prepare_map_and_path(params):
    """
    Prepare costmap and static path based on the mini env params
    :param params MiniEnvParams: params of the mini env
    :return Tuple[CostMap2D, np.ndarray(3, 2)]: costmap with obstacles and path to follow
    """

    obstacles = [
        Wall(from_pt=params.obstacle_o.as_np(),
             to_pt=params.obstacle_a.as_np()),
        Wall(from_pt=params.obstacle_o.as_np(),
             to_pt=params.obstacle_b.as_np()),
    ]

    coarse_static_path = np.array([params.start_pos.as_np(), params.end_pos.as_np()])

    static_map = CostMap2D.create_empty(
        world_size=(params.h, params.w),  # x width, y height
        resolution=params.env_params.resolution,
        world_origin=(-params.h / 2., -params.w /2.)
    )

    for obs in obstacles:
        static_map = obs.render(static_map)

    return static_map, coarse_static_path


class MiniEnv(PlanEnv):
    """
    Robotic Planning Environment in which the task
    is to run from point to point and these are not too far.
    In the environment there is only one obstacle.
    """
    def __init__(self, config):
        """
        initialize the MiniEnv
        :param config MiniEnvParams: parametrization of this environment
        """
        self._config = config
        costmap, path = prepare_map_and_path(config)
        super(MiniEnv, self).__init__(costmap, path, config.env_params)


class RandomMiniEnv(object):
    """
    MiniEnv where the env geometry is drawn randomly.

    if draw_new_turn_on_reset is True, it samples new turn
    on env.reset(), otherwise it keeps showing same turn.
    """
    def __init__(self, params=None, draw_new_turn_on_reset=True, seed=None, rng=None):
        """
        Initialize random mini environment.

        :param params RandomMiniEnvParams: parametrization of this environment.
        :param draw_new_turn_on_reset bool: should we initialize a new random environment on each reset
        :param seed int: the random seed
        :param rng np.random.RandomState: independent random state
        """
        if params is None:
            self._params = RandomMiniEnvParams(
                env_params=EnvParams(
                    goal_ang_dist=np.pi / 8.,
                    goal_spat_dist=0.2,
                )
            )
        else:
            self._params = params

        if rng is None:
            self._rng = np.random.RandomState(seed=0)
        else:
            self._rng = rng

        self.seed(seed)
        self._draw_new_turn_on_reset = draw_new_turn_on_reset

        mini_params = _sample_mini_env_params(self._params, self._rng)
        self._env = MiniEnv(mini_params)

        self.action_space = self._env.action_space

    def seed(self, seed=None):
        """ Seed the random state of the environment.
         :param seed int: the seeding constant
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
            mini_params = _sample_mini_env_params(self._params, self._rng)
            self._env = MiniEnv(mini_params)

        return self._env.reset()

    def render(self, mode='human'):
        """
        Render human-friendly representation of the environment on the screen.
        :param mode str: the mode of rendering, currently only 'human' works
        :return np.ndarray: Perhaps the image returned by the environment
        """
        return self._env.render(mode)

    def close(self):
        """ Do whatever you need to do on closing: release the resources etc. """
        self._env.close()
