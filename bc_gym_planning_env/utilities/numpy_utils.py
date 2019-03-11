"""some faster or more convenient numpy utilities"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.core.numeric import concatenate


def fast_vstack(tup):
    """
    Stack arrays in sequence vertically (row wise).
    Faster version of vstack (traditional vstack calls asanyarray too many times)
    :param tup Collection[array]: input arrays in a tuple(see vstack)
    :return array: stacked array see vstack
    """
    arrays = []
    for a in tup:
        assert isinstance(a, np.ndarray)
        # this is fast implementation of atleast_2d for arrays
        if len(a.shape) == 0:
            a = a.reshape(1, 1)
        elif len(a.shape) == 1:
            a = a[np.newaxis, :]

        arrays.append(a)
    return concatenate(arrays, 0)


def fast_hstack(tup):
    """
    Stack arrays in sequence horizontally (column wise).
    Faster version of hstack (traditional hstack calls asanyarray too many times)
    :param tup Collection[array]: input arrays in a tuple(see hstack)
    :return array: stacked array see hstack
    """
    arrays = []
    for a in tup:
        assert isinstance(a, np.ndarray)
        # this is fast implementation of atleast_1d for arrays
        if len(a.shape) == 0:
            a = a.reshape(1)
        arrays.append(a)

    # As a special case, dimension 0 of 1-dimensional arrays is "horizontal"
    if arrays[0].ndim == 1:
        return concatenate(arrays, 0)
    else:
        return concatenate(arrays, 1)


try:
    from brain.shining_utils.numpy_utils import amax_impl

    fast_amax = amax_impl
    """
    Equivalent of amax without any other features (just takes 1d array)
    :param input_array array(N)[T]: input array
    :return T: maximum of this array
    """
except ImportError:
    fast_amax = np.amax


try:
    from brain.shining_utils.numpy_utils import amin_impl
    fast_amin = amin_impl
    """
    Equivalent of amin without any other features (just takes 1d array)
    :param input_array array(N)[T]: input array
    :return T: minimum of this array
    """
except ImportError:
    fast_amin = np.amin
