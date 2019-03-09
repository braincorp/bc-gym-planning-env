""" Basic custom errors for the module"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


class RobotCollidedException(Exception):
    """ Exception that expresses robot collision """
    pass


def _default_raise_on_crash(*args):
    """
    The default exception we want to raise on robot's crash.
    :param args: Whatever, ignored
    """
    raise RobotCollidedException
