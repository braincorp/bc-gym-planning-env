""" Type representing observation returned from env.step(action) """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import attr
import numpy as np

from bc_gym_planning_env.utilities.costmap_2d import CostMap2D


@attr.s(frozen=True)
class Observation(object):
    """ Type representing observation returned from env.step(action) """
    pose = attr.ib(type=np.ndarray)        # oriented 2d pose of the robot
    path = attr.ib(type=np.ndarray)        # oriented path to follow
    costmap = attr.ib(type=CostMap2D)  # costmap showing obstacles
    time = attr.ib(type=float)             # what is the current timestamp
    dt = attr.ib(type=float)               # how much time passes between observations
    robot_state = attr.ib(factory=list)    # wheel_angle, measured_v, measured_w, steering_motor_command
