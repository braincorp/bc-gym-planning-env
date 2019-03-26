""" Objects & utils resolving whether the agent should get reward, given the state """
from __future__ import absolute_import

import attr
import numpy as np

from bc_gym_planning_env.utilities.path_tools import pose_distances, find_last_reached
from bc_gym_planning_env.utilities.serialize import Serializable
from bc_gym_planning_env.envs.base.reward_provider_examples import RewardProviderStateExamples


@attr.s(cmp=False)
class ContinuousRewardProviderState(Serializable):
    """ State of the continuous reward provider: """
    # How much progress has the agent done towards current goal pose
    min_spat_dist_so_far = attr.ib(type=float)
    # Full static path to follow
    path = attr.ib(type=np.ndarray)
    # idx of current goal pose along the static path
    target_idx = attr.ib(type=int)

    VERSION = 1
    reward_provider_state_type_name = RewardProviderStateExamples.CONTINUOUS_REWARD_STATE

    def __eq__(self, other):
        if not isinstance(other, ContinuousRewardProviderState):
            return False

        if (self.path != other.path).any():
            return False

        if self.min_spat_dist_so_far != other.min_spat_dist_so_far:
            return False

        if self.target_idx != other.target_idx:
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def copy(self):
        """ Get a copy of the reward provider's state
        :return ContinuousRewardProviderState: the copy of ourselves
        """
        return attr.evolve(self, path=np.copy(self.path))

    def current_goal_pose(self):
        """ Get the current goal pose
        :return np.ndarray(3): The current goal pose
        """
        # pylint: disable=unsubscriptable-object
        if self.target_idx < len(self.path):
            return self.path[self.target_idx]
        else:
            raise ValueError("No path left to follow.")

    def current_path(self):
        """ Get the current path
        :return np.ndarray(N, 3): the piece of static path left to follow
        """
        # pylint: disable=invalid-slice-index,unsubscriptable-object
        return self.path[self.target_idx:]

    def done(self):
        """ Are we done?
        :return bool: True if we are done with this environment. """
        return self.target_idx > len(self.path) - 1

    def get_reward_provider_state_type_name(self):
        """ Get the type (string describing the type) of reward provider state type.
        :return RewardProviderStateExamples: enum member defining type of reward provider state type
        """
        return self.reward_provider_state_type_name


@attr.s(cmp=False)
class ContinuousRewardPurePursuitProviderState(Serializable):
    """ State of the continuous reward provider: """
    # How much progress has the agent done towards current goal pose
    min_spat_dist_so_far = attr.ib(type=float)
    # Full static path to follow
    path = attr.ib(type=np.ndarray)
    # idx of current goal pose along the static path
    target_idx = attr.ib(type=int, default=0)

    VERSION = 1
    reward_provider_state_type_name = RewardProviderStateExamples.CONTINUOUS_REWARD_PURE_PURSUIT_STATE

    def __eq__(self, other):
        if not isinstance(other, ContinuousRewardProviderState):
            return False

        if (self.path != other.path).any():
            return False

        if self.min_spat_dist_so_far != other.min_spat_dist_so_far:
            return False

        if self.target_idx != other.target_idx:
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def copy(self):
        """ Get a copy of the reward provider's state
        :return ContinuousRewardProviderState: the copy of ourselves
        """
        return attr.evolve(self, path=np.copy(self.path))

    def current_goal_pose(self):
        """ Get the current goal pose
        :return np.ndarray(3): The current goal pose
        """
        return self.path[-1]

    def current_path(self):
        """ Get the current path
        :return np.ndarray(N, 3): the piece of static path left to follow
        """
        return self.path[:self.target_idx+1]

    def update_goal(self, pose, radius=2.):
        """
        search and update the front targeting point which is the first waypoint that is more than radius away
        :param pose: current pose of the robot
        :param radius: the distance between the robot and the target waypoint
        :return: the updated target waypoint's index on the path
        """
        for i in range(self.target_idx, len(self.path)):
            if np.linalg.norm(self.path[i, :2] - pose[:2]) > radius:
                self.target_idx = i
                return self.target_idx
        self.target_idx = len(self.path) - 1
        return self.target_idx  # if no valid point, return last point

    def done(self, state):
        """ Are we done?
        :param state: current state of the environment
        :return bool: True if we are done with this environment. """
        robot_pose = state.pose
        spat_dist, _ = pose_distances(self.current_goal_pose(), robot_pose)
        spat_near = spat_dist < 1.0
        goal_reached = spat_near

        return goal_reached

    def get_reward_provider_state_type_name(self):
        """ Get the type (string describing the type) of reward provider state type.
        :return RewardProviderStateExamples: enum member defining type of reward provider state type
        """
        return self.reward_provider_state_type_name


@attr.s
class RewardParams(Serializable):
    """ Parametrization of the continuous reward provider"""
    VERSION = 1
    # How close spatially do you have to be to count the goal as reached
    spatial_precision = attr.ib(type=float)
    # How close angularly do you have to be to count the goal as reached
    angular_precision = attr.ib(type=float)
    # How much reward to assign when you are progressing toward the current goal
    spatial_progress_multiplier = attr.ib(type=float, default=0.0)


@attr.s
class ContinuousRewardProvider(object):
    """ An object resolving whether the agent should get reward.
    Gives reward of 1.0 when we reach any waypoint.
    If we reach any waypoint, our goal is the next waypoint.
     """
    _params = attr.ib(type=RewardParams)
    _state = attr.ib(type=ContinuousRewardProviderState, default=None)
    _initial_state = attr.ib(type=ContinuousRewardProviderState, default=None)

    def set_state(self, state):
        """ Set the state of the environment
        :param state ContinuousRewardProviderState: the state of the reward provider
        """
        if self._initial_state is None:
            self._initial_state = state.copy()
        self._state = state.copy()

    def get_state(self):
        """ Get the state of the reward provider
        :return ContinuousRewardProviderState: the state of the reward provider
        """
        return self._state.copy()

    def get_current_path(self):
        """ Get the current piece of the static path
        :return np.ndarray: The piece of the static path left to follow
        """
        return self._state.current_path()

    def done(self, state):  # pylint: disable=unused-argument
        """
        Are there any more goals to accomplish?
        Enviornment can have more criteria for finishing the episode,
        e.g. getting out of bounds etc.
        :param state State: not used in this version of the reward system, but put here for consistent API interface
        :return float: whether this episode is finished or not
        """
        return self._state.done()

    def reward(self, state):
        """
        If you have reached any point, then 1.
        Otherwise if you are closer to the next waypoint then you used to be,
        get some small reward.

        :param state State: full state of the environment
        :return float: the reward
        """
        if self._state.done():
            return 0.0

        # See if you have reached any new point
        last_reached_idx = find_last_reached(
            state.pose,
            self._state.path,
            self._params.spatial_precision,
            self._params.angular_precision
        )

        if last_reached_idx is not None and last_reached_idx >= self._state.target_idx:
            # assign the reward for reaching a waypoint
            next_idx = last_reached_idx + 1

            self._state.target_idx = next_idx

            # set up the new waypoint
            if not self._state.done():
                dist_to_goal, _ = pose_distances(self._state.current_goal_pose(), state.pose)
                self._state.min_spat_dist_so_far = dist_to_goal
            else:
                self._state.min_spat_dist_so_far = 0.0

            return 1.0
        else:
            # maybe we get some progress towards current waypoint
            dist_to_goal, _ = pose_distances(self._state.current_goal_pose(), state.pose)

            if dist_to_goal < self._state.min_spat_dist_so_far:
                # We progressed toward the current waypoint
                reward = self._state.min_spat_dist_so_far - dist_to_goal
                self._state.min_spat_dist_so_far = dist_to_goal
                return reward * self._params.spatial_progress_multiplier
            else:
                # We didn't do any progress
                return 0.0

    @staticmethod
    def generate_initial_state(path, params):
        """ Generate the initial state of the reward provider.
        :param path np.ndarray(N, 3): the static path
        :param params RewardParams: parametrization of the reward provider
        :return ContinuousRewardProviderState: the initial state of the reward provider
        """
        initial_pose = path[0]
        last_reached_idx = find_last_reached(
            initial_pose, path,
            params.spatial_precision,
            params.angular_precision
        )

        if last_reached_idx == len(path) - 1:
            # We are out of path to follow at the beginning!
            raise ValueError("Goal pose too close to initial pose")
        else:
            target_idx = last_reached_idx + 1

        goal_pose = path[target_idx]
        dist_to_goal, _ = pose_distances(goal_pose, initial_pose)

        return ContinuousRewardProviderState(
            min_spat_dist_so_far=dist_to_goal,
            path=path,
            target_idx=target_idx
        )


@attr.s
class ContinuousRewardPurePursuitProvider(object):
    """ An object resolving whether the agent should get reward.
    Gives reward of 1.0 when we reach any waypoint.
    If we reach any waypoint, our goal is the next waypoint.
     """
    _params = attr.ib(type=RewardParams)
    _state = attr.ib(type=ContinuousRewardPurePursuitProviderState, default=None)
    _initial_state = attr.ib(type=ContinuousRewardPurePursuitProviderState, default=None)

    def set_state(self, state):
        """ Set the state of the environment
        :param state ContinuousRewardPurePursuitProviderState: the state of the reward provider
        """
        if self._initial_state is None:
            self._initial_state = state.copy()
        self._state = state.copy()

    def get_state(self):
        """ Get the state of the reward provider
        :return ContinuousRewardPurePursuitProviderState: the state of the reward provider
        """
        return self._state.copy()

    def get_current_path(self):
        """ Get the current piece of the static path
        :return np.ndarray: The piece of the static path left to follow
        """
        return self._state.current_path()

    def done(self, state):
        """
        Are there any more goals to accomplish?
        Enviornment can have more criteria for finishing the episode,
        e.g. getting out of bounds etc.
        :param state: current state of the environment
        :return float: whether this episode is finished or not
        """
        return self._state.done(state)

    def reward(self, state):
        """
        If you have reached any point, then 1.
        Otherwise if you are closer to the next waypoint then you used to be,
        get some small reward.

        :param state State: full state of the environment
        :return float: the reward
        """

        self._state.update_goal(state.pose)

        reward = -0.05

        dist_to_goal, _ = pose_distances(self._state.current_goal_pose(), state.pose)
        reward += self._state.min_spat_dist_so_far - dist_to_goal
        self._state.min_spat_dist_so_far = dist_to_goal

        if state.robot_collided:
            reward -= 100

        return reward

    @staticmethod
    def generate_initial_state(path, params):   # pylint: disable=unused-argument
        """ Generate the initial state of the reward provider.
        :param path np.ndarray(N, 3): the static path
        :param params RewardParams: parametrization of the reward provider, not used here but kept it for consistent API call
        :return ContinuousRewardProviderState: the initial state of the reward provider
        """
        initial_pose = path[0]

        target_idx = 0
        goal_pose = path[-1]
        dist_to_goal, _ = pose_distances(goal_pose, initial_pose)

        return ContinuousRewardPurePursuitProviderState(
            min_spat_dist_so_far=dist_to_goal,
            path=path,
            target_idx=target_idx
        )
