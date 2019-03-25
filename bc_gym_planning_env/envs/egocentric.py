""" Classes & utils for extracting egocentric observations from the planning environment.
The observation is a colored egocentric costmap and a vector with (robot state, normalized goal coordinates).
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from collections import OrderedDict

import numpy as np

from bc_gym_planning_env.envs.base import spaces
from bc_gym_planning_env.utilities.coordinate_transformations import from_global_to_egocentric, world_to_pixel
from bc_gym_planning_env.utilities.costmap_utils import extract_egocentric_costmap


class ObservationWrapper(object):
    """ A generic abstract observation wrapper. Override .observation() method in the subclass."""
    def __init__(self, env):
        """ Intitialize the observation wrapper.
        :param env object: the environment to wrap.
        """
        self.env = env
        self.action_space = self.env.action_space

    def unwrapped(self):
        """ Unwrap the env from this wrapper.
        :return object: the resulting environment
        """
        return self.env

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
        :return object: whatever the type of observation this wrapper is returning
        """
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Resets the 'done' state as well.

        :return object: whatever the type of observation this wrapper is returning
        """
        observation = self.env.reset()
        return self.observation(observation)

    def observation(self, observation):
        """
        The method that should be overriden in the subclass.
        This method converts from one observation type to another type.
        For example, we could convert the state of the whole environment to egocentric
        costmap.
        :param observation Observation: the observation coming from the environment
        it should return the augmented observation
        """
        raise NotImplementedError

    def render(self, mode='human'):
        """
        Render human-friendly representation of the environment on the screen.
        :param mode str: the mode of rendering, currently only 'human' works
        :return np.ndarray: Perhaps the image returned by the environment
        """
        return self.env.render(mode)

    def close(self):
        """ Do whatever you need to do on closing: release the resources etc. """
        if self.env:
            self.env.close()

    def seed(self, seed=None):
        """ Seed the environment's random state.
         :param seed int: the seed value
         """
        self.env.seed(seed)

    def get_state(self):
        """ Get the state of the base environment underlying this simulation.
        :return State: the state of the underlying base environment
        """
        return self.env.get_state().copy()

    def set_state(self, state):
        """ Set the state of the environment
        :param state State: the state of the base planning environment
        """
        self.env.set_state(state)


class EgocentricCostmap(ObservationWrapper):
    """ Random Aisle Turn environment, but the observation is a colored egocentric
    costmap and a vector with (robot state, normalized goal coordinates). """

    def __init__(self, env):
        """
        Wrap the environment in this wrapper, that will make the observation egocentric
        :param env object: the environment to wrap.
        """
        super(EgocentricCostmap, self).__init__(env)
        # As openai gym style requires knowing resolution of the image up front
        self._egomap_x_bounds = np.array([-0.5, 3.])  # aligned with robot's direction
        self._egomap_y_bounds = np.array([-2., 2.])  # orthogonal to robot's direction
        resulting_size = np.array([self._egomap_x_bounds[1] - self._egomap_x_bounds[0],
                                   self._egomap_y_bounds[1] - self._egomap_y_bounds[0]])

        pixel_size = world_to_pixel(resulting_size, np.zeros((2,)), resolution=0.03)
        data_shape = (pixel_size[1], pixel_size[0], 1)
        self.observation_space = spaces.Dict(OrderedDict((
            ('env', spaces.Box(low=0, high=255, shape=data_shape, dtype=np.uint8)),
            ('goal', spaces.Box(low=-1., high=1., shape=(3, 1), dtype=np.float64))
        )))

    def observation(self, observation):
        """Extract egocentric map and path from rich observation
        :param observation Observation: observation, generated by the non-wrapped class
        :return (array(W, H)[uint8], array(N, 3)[float]): egocentric obstacle data and a path
        """
        costmap = observation.costmap
        robot_pose = observation.pose

        ego_costmap = extract_egocentric_costmap(
            costmap,
            robot_pose,
            resulting_origin=(self._egomap_x_bounds[0], self._egomap_y_bounds[0]),
            resulting_size=np.array([self._egomap_x_bounds[1] - self._egomap_x_bounds[0],
                                     self._egomap_y_bounds[1] - self._egomap_y_bounds[0]])
        )

        ego_path = from_global_to_egocentric(observation.path, robot_pose)
        obs = np.expand_dims(ego_costmap.get_data(), -1)
        if len(ego_path) == 0:
            # 3 is goal len + 6 is len of the state
            size_of_goal_n_state = 3 + observation.robot_state.to_numpy_array().shape[0]
            return OrderedDict((
                ('env', obs),
                # 9 values characterize robot's state including velocities
                ('goal_n_state', np.expand_dims(np.zeros(size_of_goal_n_state, dtype=np.float32), -1))
            ))
        else:
            normalized_goal = ego_path[0, :2] / ego_costmap.world_size()
            normalized_goal = np.clip(normalized_goal, (-1., -1.), (1., 1.))
            goal_pose = np.hstack([normalized_goal, ego_path[0, 2]])
            goal_n_state = np.hstack([goal_pose, observation.robot_state.to_numpy_array()])
            # observation.robot_state is: wheel_angle, measured_v, measured_w, steering_motor_command
            return OrderedDict((
                ('env', obs),
                ('goal_n_state', np.expand_dims(goal_n_state, -1).astype(np.float32))
             ))
