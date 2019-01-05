from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


class RobotCollidedException(Exception):
    pass


def _default_raise_on_crash(*args):
    raise RobotCollidedException
