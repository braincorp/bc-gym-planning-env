""" Code for wrapping the motion primitive action in an object. """
from __future__ import division
from __future__ import absolute_import

import attr
import numpy as np

from bc_gym_planning_env.utilities.serialize import Serializable


@attr.s(cmp=False)
class Action(Serializable):
    """ Object representing an 'action' - a motion primitive to execute in the environment """
    VERSION = 1
    command = attr.ib(type=np.ndarray)

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False

        if (self.command != other.command).any():
            return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)
