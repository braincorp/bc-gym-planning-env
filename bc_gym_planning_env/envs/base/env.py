""" Robot planning problem turned into openai gym-like, reinforcement learning style environment """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import attr
import copy
import numpy as np
from bc_gym_planning_env.robot_models.tricycle_model import TricycleRobot
from bc_gym_planning_env.robot_models.robot_dimensions_examples import get_dimensions_example
from bc_gym_planning_env.robot_models.robot_examples_factory import create_standard_robot
from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.utilities.serialize import Serializable
from bc_gym_planning_env.utilities.costmap_utils import clone_costmap
from bc_gym_planning_env.utilities.coordinate_transformations import world_to_pixel
from bc_gym_planning_env.utilities.path_tools import get_pixel_footprint
from bc_gym_planning_env.utilities.path_tools import refine_path
from bc_gym_planning_env.envs.base.draw import draw_environment
from bc_gym_planning_env.envs.base.obs import Observation
from bc_gym_planning_env.envs.base.params import EnvParams
from bc_gym_planning_env.envs.base import spaces
from bc_gym_planning_env.envs.base.reward_provider_examples_factory import\
    create_reward_provider_state, get_reward_provider_example
from bc_gym_planning_env.utilities.gui import OpenCVGui


def _get_element_from_list_with_delay(item_list, element, delay):
    """
    A little util for faking delay of data stream. e.g.
    ```
    l = []
    get = generate_delay(l, 3)

    for i in range(10):
        print get(i)
    ```
    prints
    0 0 0 1 2 3 4 5 6

    :param item_list list: list of items
    :param element object: Just any python object
    :param delay int: how many items to delay by
    :return: a function that fakes a delay data stream, see above
    """
    item_list.append(element)
    if len(item_list) > delay:
        return item_list.pop(0)
    else:
        return item_list[0]


@attr.s(cmp=False)
class State(Serializable):
    """ State of the environemnt that you can reset your environment to.
    However, it doesn't contain parametrization. """
    reward_provider_state = attr.ib(type=object)
    path = attr.ib(type=np.ndarray)
    original_path = attr.ib(type=np.ndarray)
    costmap = attr.ib(type=CostMap2D)
    iter_timeout = attr.ib(type=int)
    current_time = attr.ib(type=float)
    current_iter = attr.ib(type=int)
    robot_collided = attr.ib(type=bool)
    poses_queue = attr.ib(type=list)
    robot_state_queue = attr.ib(type=list)
    control_queue = attr.ib(type=list)
    pose = attr.ib(type=np.ndarray)
    robot_state = attr.ib(type=object)

    VERSION = 1

    def copy(self):
        """ Get the copy of the environment.
        :return State: get the state of the environment
        """
        # pylint: disable=no-member
        return attr.evolve(
            self,
            reward_provider_state=self.reward_provider_state.copy(),
            path=np.copy(self.path),
            pose=np.copy(self.pose),
            original_path=np.copy(self.original_path),
            costmap=clone_costmap(self.costmap),
            poses_queue=copy.deepcopy(self.poses_queue),
            robot_state_queue=copy.deepcopy(self.robot_state_queue),
            control_queue=copy.deepcopy(self.control_queue),
            robot_state=self.robot_state.copy()
        )

    def __eq__(self, other):
        # pylint: disable=too-many-return-statements
        if not isinstance(other, State):
            return False

        if self.reward_provider_state != other.reward_provider_state:
            return False

        if (self.path != other.path).any():
            return False

        if (self.original_path != other.original_path).any():
            return False

        if self.costmap != other.costmap:
            return False

        if self.iter_timeout != other.iter_timeout:
            return False

        if self.current_time != other.current_time:
            return False

        if self.current_iter != other.current_iter:
            return False

        if self.robot_collided != other.robot_collided:
            return False

        if self.poses_queue != other.poses_queue:
            return False

        if self.robot_state_queue != other.robot_state_queue:
            return False

        if self.control_queue != other.control_queue:
            return False

        if (self.pose != other.pose).any():
            return False

        if self.robot_state != other.robot_state:
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def deserialize(cls, state):

        ver = state.pop('version')
        assert ver == cls.VERSION

        state['costmap'] = CostMap2D.from_state(state['costmap'])

        reward_provider_state_instance = create_reward_provider_state(state.pop('reward_provider_state_name'))
        state['reward_provider_state'] = reward_provider_state_instance.deserialize(state['reward_provider_state'])

        # prepare for robot state deserialization
        robot_instance = create_standard_robot(state.pop('robot_type_name'))
        robot_state_type = robot_instance.get_state_type()

        # deserialize the robot state
        state['robot_state'] = robot_state_type.deserialize(state['robot_state'])

        # deserialize robot state queue
        acc = []
        for item in state['robot_state_queue']:
            acc.append(robot_state_type.deserialize(item))
        state['robot_state_queue'] = acc

        return cls(**state)

    def serialize(self):
        resu = attr.asdict(self)

        # pylint: disable=no-member
        resu['version'] = self.VERSION
        resu['costmap'] = self.costmap.get_state()
        resu['reward_provider_state_type_name'] = self.reward_provider_state.get_reward_provider_state_type_name()
        resu['reward_provider_state'] = self.reward_provider_state.serialize()
        resu['robot_type_name'] = self.robot_state.get_robot_type_name()
        resu['robot_state'] = self.robot_state.serialize()

        return resu


def make_initial_state(path, costmap, robot, reward_provider, params):
    """ Prepare the initial full state of the planning environment
    :param path: the static path to follow
    :param costmap: the static costmap containg all the obstacles
    :param robot: robot - we will execute the motion based on its model
    :param reward_provider: an instance of the reward computing class
    :param params: parametriztion of the environment
    :return State: the full initial state of the environment
    """

    if params.refine_path:
        path = refine_path(path, params.path_delta)

    assert path.shape[1] == 3

    # generate robot_state, poses,
    initial_pose = path[0]
    robot_state = robot.get_initial_state()
    robot_state.set_pose(initial_pose)

    return State(
        reward_provider_state=reward_provider.generate_initial_state(path, params.reward_provider_params),
        path=np.ascontiguousarray(path),
        original_path=np.copy(np.ascontiguousarray(path)),
        costmap=costmap,
        iter_timeout=params.iteration_timeout,
        current_time=0.0,
        current_iter=0,
        robot_collided=False,
        pose=initial_pose,
        poses_queue=[],
        robot_state=robot_state,
        robot_state_queue=[],
        control_queue=[],
    )


class PlanEnv(Serializable):
    """ Poses planning problem as OpenAI gym task. """
    def __init__(self, costmap, path, params):
        """
        :param costmap CostMap2D: costmap denoting obstacles
        :param path array(N, 3): oriented path, presented as way points
        :param params EnvParams: parametrization of the environment
        """
        # Stateful things
        self._robot = TricycleRobot(dimensions=get_dimensions_example(params.robot_name))
        reward_provider_example = get_reward_provider_example(params.reward_provider_name)
        self._reward_provider = reward_provider_example(params=params.reward_provider_params)

        # Properties, things without state
        self.action_space = spaces.Discrete(11)
        self.reward_range = (0.0, 1.0)
        self._gui = OpenCVGui()
        self._params = params

        # State
        self._state = make_initial_state(path, costmap, self._robot, self._reward_provider, params)
        self._initial_state = self._state.copy()

        self.set_state(self._state)

    def serialize(self):

        serialized = {
            'version': self.VERSION,
            'state': self._state.serialize(),
            'params': self._params.serialize(),
            'path': self._state.original_path,
            'costmap': self._state.costmap.get_state()
        }

        return serialized

    @classmethod
    def deserialize(cls, state):

        ver = state.pop('version')
        assert ver == cls.VERSION

        init_costmap = CostMap2D.from_state(state['costmap'])
        init_path = state['path']
        params = EnvParams.deserialize(state['params'])
        state = State.deserialize(state['state'])
        instance = cls(init_costmap, init_path, params)
        instance.set_state(state)

        return instance

    def set_state(self, state):
        """ Set the state of the environment
        :param state State: State of the environment to set the env to
        """
        state = state.copy()
        self._state = state
        self._robot.set_state(self._state.robot_state)
        self._reward_provider.set_state(self._state.reward_provider_state)

    def get_state(self):
        """ Get current state (but not parametrization) of the environment
        :return State: the state of the environment
        """
        return self._state.copy()

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Resets the 'done' state as well.

        :return Observation:  observation on reset of the environment,
                              to be fed to agent as the initial observation.
        """

        self.set_state(self._initial_state)
        return self._extract_obs()

    def render(self, mode='human'):
        """
        Render human-friendly representation of the environment on the screen.
        :param mode str: the mode of rendering, currently only 'human' works
        :return np.ndarray: the human-friendly image representation returned by the environment
        """

        if mode not in ['human', 'rgb_array']:
            raise NotImplementedError

        img = draw_environment(self._state.path, self._state.original_path, self._robot, self._state.costmap)

        if mode == 'human':
            return self._gui.display(img)
        else:
            return img

    def close(self):
        """ Do whatever you need to do on closing: release the resources etc. """
        self._gui.close()

    def seed(self, seed=None):
        """ Seeding actually doesn't do on the level of this environment,
        as it should be fully deterministic. The environments deriving or
         using this class it might do something here
         :param seed object: whatever you want to use for seeding
         """
        pass

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

        # Process the environment dynamics
        self._state = self._resolve_state_transition(action, self._state)

        reward = self._reward_provider.reward(self._state)
        obs = self._extract_obs()
        info = self._extract_info()
        done = self._extract_done(self._state)

        self._state.reward_provider_state = self._reward_provider.get_state()
        self._state.path = self._reward_provider.get_current_path()

        return obs, reward, done, info

    def _resolve_state_transition(self, action, state):
        """
        Mutate state of the environment based on the received motion command.
        :param action Tuple[float, float]: motion command (wheel_v, wheel_angle)
        :param state State: current state of the environment
        :return State: the state of the environment after application of the transition function
        """

        delayed_action = _get_element_from_list_with_delay(
            state.control_queue, action, self._params.control_delay
        )

        collided = _env_step(self._state.costmap, self._robot, self._params.dt, delayed_action)

        pose = self._robot.get_pose()
        delayed_pose = _get_element_from_list_with_delay(
            state.poses_queue, pose, self._params.pose_delay
        )

        current_time = state.current_time + self._params.dt
        current_iter = state.current_iter + 1

        robot_state = self._robot.get_state()
        delayed_robot_state = _get_element_from_list_with_delay(
            state.robot_state_queue, robot_state, self._params.state_delay

        )

        state.current_time = current_time
        state.current_iter = current_iter
        state.robot_collided = state.robot_collided or collided
        state.pose = delayed_pose
        state.path = self._reward_provider.get_current_path()
        state.robot_state = delayed_robot_state

        return state

    def _has_timed_out(self):
        """
        Has the environment timed out?
        :return bool: Has the environment timed out?
        """
        return self._state.current_iter >= self._params.iteration_timeout

    def _extract_done(self, state):
        """
        Extract if we are done with this enviroment.
        For example we are done, if the goal has been reached,
        we have timed out or the robot has collided.
        :param state: current state of the environment
        :return bool: are we done with this planning environment?
        """
        goal_reached = self._reward_provider.done(state)
        timed_out = self._has_timed_out()
        done = goal_reached or timed_out or self._state.robot_collided

        return done

    def _extract_obs(self):
        """
        Extract an observation from the environment.
        :return Observation: the observation to process
        """
        return Observation(
            pose=self._state.pose,
            path=self._state.path,
            costmap=self._state.costmap,
            robot_state=self._state.robot_state,
            time=self._state.current_time,
            dt=self._params.dt
        )

    @staticmethod
    def _extract_info():
        """ Extract debug information from the env. For now empty.
        :return Dict: empty dict (for now) """
        return {}


def _env_step(costmap, robot, dt, control_signals):
    """
    Execute movement step for the robot.
    :param costmap Costmap2D: costmap containing the obstacles to potentially collide with
    :param robot: Robot that will execute the movement based on its model
    :param dt: time interval between time steps
    :param control_signals: motion primitives to executed
    :return bool: Does it collide?
    """

    old_position = robot.get_pose()
    robot.step(dt, control_signals)
    new_position = robot.get_pose()

    x, y, angle = new_position
    collides = pose_collides(x, y, angle, robot, costmap)
    if collides:
        robot.set_pose(*old_position)

    return collides


def pose_collides(x, y, angle, robot, costmap):
    """
    Check if robot footprint at x, y (world coordinates) and
        oriented as yaw collides with lethal obstacles.
    :param x: robot pose
    :param y: robot pose
    :param angle: robot pose
    :param robot: Robot that will supply the footprint
    :param costmap Costmap2D: costmap containing the obstacles to collide with
    :return bool : does the pose collide?
    """
    kernel_image = get_pixel_footprint(angle, robot.get_footprint(), costmap.get_resolution())
    # Get the coordinates of where the footprint is inside the kernel_image (on pixel coordinates)
    kernel = np.where(kernel_image)
    # Move footprint to (x,y), all in pixel coordinates
    x, y = world_to_pixel(np.array([x, y]), costmap.get_origin(), costmap.get_resolution())
    collisions = y + kernel[0] - kernel_image.shape[0] // 2, x + kernel[1] - kernel_image.shape[1] // 2
    raw_map = costmap.get_data()
    # Check if the footprint pixel coordinates are valid, this is, if they are not negative and are inside the map
    good = np.logical_and(np.logical_and(collisions[0] >= 0, collisions[0] < raw_map.shape[0]),
                          np.logical_and(collisions[1] >= 0, collisions[1] < raw_map.shape[1]))

    # Just from the footprint coordinates that are good, check if they collide
    # with obstacles inside the map
    return bool(np.any(raw_map[collisions[0][good],
                               collisions[1][good]] == CostMap2D.LETHAL_OBSTACLE))
