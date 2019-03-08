from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import attr
from gym import spaces, Env
import numpy as np

from bc_gym_planning_env.envs.internals.error import RobotCollidedException
from bc_gym_planning_env.envs.internals.error import _default_raise_on_crash
from bc_gym_planning_env.envs.internals.obs import Observation
from bc_gym_planning_env.utilities.gui import OpenCVGui
from bc_gym_planning_env.utilities.path_tools import limit_path_index, find_last_reached, pose_distances
from bc_gym_planning_env.utilities.planning_environment import PlanningEnvironment
from bc_gym_planning_env.robot_models.standard_robot_names_examples import StandardRobotExamples
from bc_gym_planning_env.robot_models.tricycle_model import TricycleRobot


def generate_delay(item_list, delay):
    """ A little util for faking delay of data stream. e.g.
    ```
    l = []
    get = generate_delay(l, 3)

    for i in range(10):
        print get(i)
    ```
    prints
    0 0 0 1 2 3 4 5 6
    """
    def get_element_from_list_with_delay(element):
        item_list.append(element)
        if len(item_list) > delay:
            return item_list.pop(0)
        else:
            return item_list[0]

    return get_element_from_list_with_delay


@attr.s(frozen=True)
class EnvParams(object):
    """ Parametrization of the environment.  """
    control_delay = attr.ib(default=0, type=int)               # how delayed is the mapping between
    dt = attr.ib(type=float, default=0.05)                     # how much time passes between two observations
    goal_ang_dist = attr.ib(type=float, default=np.pi/2)       # how close angularily to goal to reach it
    goal_spat_dist = attr.ib(type=float, default=1.0)          # how close to goal to reach it
    initial_wheel_angle = attr.ib(default=0.0, type=float)     # initialization of wheel angle
    iteration_timeout = attr.ib(type=int, default=1200)        # how many timesteps to reach the goal
    path_limiter_angular_precision = attr.ib(type=float,       # angular precision of path follower
                                             default=np.pi/4)
    path_limiter_max_dist = attr.ib(type=float, default=5.0)   # spatial horizon of path follower
    path_limiter_spatial_precision = attr.ib(type=float,       # spatial precision of path follower
                                             default=1.0)
    pose_delay = attr.ib(default=0, type=int)                  # we perceive poses with how much delay
    robot_name = attr.ib(
        default=StandardRobotExamples.INDUSTRIAL_TRICYCLE_V1)             # name of the robot, e.g. determines footprint
    state_delay = attr.ib(default=0, type=int)                 # state perception delay
                                                               # real world delay is about 0.11s (2 steps)


class PlanEnv(Env):
    """ Poses planning problem as OpenAI gym task.
    """
    def __init__(self, costmap, path, params):
        """
        :param costmap CostMap2D: costmap denoting obstacles
        :param path array(N, 3): oriented path, presented as points
        :param params EnvParams: parametrization of the environment
        """
        robot = TricycleRobot(robots_type_name=params.robot_name)

        assert(path.shape[1] == 3)

        self.action_space = spaces.Box(
            low=np.array([
                -robot.get_max_front_wheel_speed() / 2,
                -np.pi/2
            ]),
            high=np.array([
                robot.get_max_front_wheel_speed(),
                np.pi/2
            ]),
            dtype=np.float32
        )

        self.reward_range = (0.0, 1.0)

        self._environment = PlanningEnvironment(
            costmap=costmap,
            robot=robot,
            on_collision_callback=_default_raise_on_crash
        )
        self._gui = OpenCVGui()

        self._path = np.ascontiguousarray(path)
        self._original_path = np.copy(self._path)
        self._costmap = costmap
        self._robot = robot
        self._params = params

        self._goal = path[-1]
        self._current_time = 0.0
        self._iter_timeout = params.iteration_timeout
        self._current_iter = 0
        self._robot_collided = False
        self._has_been_reset = False

        self._original_path_len = len(path)
        self._traversed_so_far = 0

        self._poses_queue = []
        self._process_pose = generate_delay(self._poses_queue, self._params.pose_delay)

        self._robot_state_queue = []
        self._process_robot_state = generate_delay(self._robot_state_queue, self._params.state_delay)

        self._control_queue = []
        self._process_command = generate_delay(self._control_queue, self._params.control_delay)

        self._pose = None
        self._robot_state = None

    def reset(self):
        self._has_been_reset = True
        self._robot.set_front_wheel_angle(self._params.initial_wheel_angle)
        self._robot_state = self._robot.get_robot_state()

        self._pose = self._process_pose(self._path[0])
        self._environment.set_robot_pose(*self._pose)

        self._path = self._process_path(self._path)

        return self._extract_obs()

    def render(self, mode='human'):
        img = self._environment.draw(self._path, self._original_path)
        self._gui.display(img)

    def close(self):
        self._gui.close()

    def seed(self, seed=None):
        # Seeding doesn't do anything here
        pass

    def step(self, motion_command):
        """ motion_command: (wheel_v, wheel_angle) for robot model """
        assert self._has_been_reset, "Need to reset the env before running."

        self._act_on_motion_commands(motion_command)
        self._process_iter_timeout()

        obs = self._extract_obs()
        reward = self._extract_reward()
        info = self._extract_info()
        done = self._extract_done()

        return obs, reward, done, info

    def _process_iter_timeout(self):
        self._current_iter += 1

    def _process_path(self, path):
        """Cut down the parts of the path that have been already traversed by local planner."""
        # TODO: Looks like this doesn't work too well. It catches waypoints only once in a while.

        if len(path):

            final_static_path_idx = limit_path_index(
                path,
                max_dist=self._params.path_limiter_max_dist
            )

            reached_idx = find_last_reached(
                pose=self._pose,
                segment=path[:final_static_path_idx],
                spatial_precision=self._params.path_limiter_spatial_precision,
                angular_precision=self._params.path_limiter_angular_precision
            )

            if reached_idx is not None:
                path = path[reached_idx+1:]
                path = np.ascontiguousarray(path)

        return path

    def _act_on_motion_commands(self, command):
        """ Mutate state based on received motion command. """
        processed_command = self._process_command(command)
        try:
            self._environment.step(self._params.dt,
                                   processed_command,
                                   check_collisions=True)
        except RobotCollidedException:
            self._robot_collided = True

        self._current_time += self._params.dt

        self._pose = self._process_pose(self._environment.get_robot_pose())
        self._robot_state = self._process_robot_state(self._robot.get_robot_state())
        self._path = self._process_path(self._path)

    def _is_goal_reached(self):
        robot_pose = self._environment.get_robot_pose()
        spat_dist, ang_dist = pose_distances(self._goal, robot_pose)
        spat_near = spat_dist < self._params.goal_spat_dist
        ang_near = ang_dist < self._params.goal_ang_dist
        goal_reached = spat_near and ang_near

        return goal_reached

    def _has_timed_out(self):
        return self._current_iter >= self._iter_timeout

    def _extract_done(self):
        goal_reached = self._is_goal_reached()
        timed_out = self._has_timed_out()
        done = goal_reached or timed_out or self._robot_collided

        return done

    def _extract_reward(self):
        """Just one example of reward: Get full reward if the goal is reached."""
        if self._is_goal_reached():
            return 1.0
        else:
            return 0.0

    def _extract_obs(self):
        return Observation(
            pose=self._pose,
            path=self._path,
            costmap=self._costmap,
            robot_state=self._robot_state,
            time=self._current_time,
            dt=self._params.dt
        )

    def _extract_info(self):
        return {}
