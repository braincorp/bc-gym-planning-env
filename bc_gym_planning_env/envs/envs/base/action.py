""" Code for wrapping the motion primitive action in an object. """
from __future__ import division
from __future__ import absolute_import

import attr
import numpy as np


@attr.s
class Action(object):
    """ Object representing an 'action' - a motion primitive to execute in the environment """
    command = attr.ib(type=np.ndarray)
