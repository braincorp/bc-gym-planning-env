""" Type representing observation returned from env.step(action) """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import attr
import numpy as np

from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.robot_models.robot_examples_factory import create_standard_robot
from bc_gym_planning_env.utilities.serialize import Serializable


@attr.s(frozen=True, cmp=False)
class Observation(Serializable):
    """ Type representing observation returned from env.step(action) """
    pose = attr.ib(type=np.ndarray)                 # oriented 2d pose of the robot
    path = attr.ib(type=np.ndarray, repr=False)     # oriented path to follow
    costmap = attr.ib(type=CostMap2D)               # costmap showing obstacles
    time = attr.ib(type=float)                      # what is the current timestamp
    dt = attr.ib(type=float)                        # how much time passes between observations
    robot_state = attr.ib(type=object)              # wheel_angle, measured_v, measured_w, steering_motor_command
    VERSION = 1

    @classmethod
    def deserialize(cls, state):
        ver = state.pop('version')
        assert ver == cls.VERSION

        state['costmap'] = CostMap2D.from_state(state['costmap'])
        robot_instance = create_standard_robot(state.pop('robot_type_name'))
        robot_state_type = robot_instance.get_state_type()

        # deserialize the robot state
        state['robot_state'] = robot_state_type.deserialize(state['robot_state'])
        return cls(**state)

    def serialize(self):
        # pylint: disable=no-member

        resu = attr.asdict(self)
        resu['version'] = self.VERSION
        resu['robot_state'] = self.robot_state.serialize()
        resu['robot_type_name'] = self.robot_state.get_robot_type_name()
        resu['costmap'] = self.costmap.get_state()

        return resu

    def descriptive_comparison(self, other):
        """ Prepare a description of how we are different from the other experiment.
        :param other Observation: the other observation to compare ourselves to.
        :return str: a multiline description of how this object is different from the other
        """
        are_identical = True
        comparison = []

        if not isinstance(other, Observation):
            comparison.append("The other object is not Observation, it is {}".format(type(other)))
            are_identical = False

        if not (self.pose == other.pose).all():
            are_identical = False
            comparison.append(
                "Poses are different -> self has \n {}, \n where other has \n {}".format(self.pose, other.pose)
            )

        if len(self.path) != len(other.path):
            are_identical = False
            comparison.append("Paths have different lengths: {} vs {}".format(len(self.path), len(other.path)))
        elif not (self.path == other.path).all():
            are_identical = False
            diff_idx, _ = np.where(self.path != other.path)
            diff_idx = list(diff_idx)
            comparison.append("Paths are different at indices {}".format(diff_idx))

        if self.costmap != other.costmap:
            are_identical = False
            comparison.append("Costmaps are different!")

        if self.time != other.time:
            are_identical = False
            comparison.append("time is different! -> {} vs {}".format(self.time, other.time))

        if self.dt != other.dt:
            are_identical = False
            comparison.append("dt is different! -> {} vs {}".format(self.dt, other.dt))

        if self.robot_state != other.robot_state:
            are_identical = False
            comparison.append(
                "robot_state is different! -> \n {} \n vs \n{}".format(self.robot_state, other.robot_state)
            )

        if are_identical:
            comparison.append("Observations are identical.")

        return comparison

    def __eq__(self, other):
        if not isinstance(other, Observation):
            return False

        if not (self.pose == other.pose).all():
            return False

        if not (self.path == other.path).all():
            return False

        if self.costmap != other.costmap:
            return False

        if self.time != other.time:
            return False

        if self.dt != other.dt:
            return False

        if self.robot_state != other.robot_state:
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)
