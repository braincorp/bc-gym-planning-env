""" Simple file utils """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os


def mkdir_p(dirname):
    """
    Check if directory exists and if not, make it.
    :param dirname: dir to create / check
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)
