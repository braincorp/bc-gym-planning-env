"""Utilities for immutable numpy arrays"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


def freeze_array(array):
    """
    Make numpy array immutable
    :param array: numpy array
    :return: immutable numpy array
    """
    array.flags.writeable = False
    return array
