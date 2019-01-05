from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os

def mkdir_p(dir):
    """ Check if directory exists and if not, make it."""
    if not os.path.exists(dir):
        os.makedirs(dir)
