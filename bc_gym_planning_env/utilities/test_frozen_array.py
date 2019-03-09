from __future__ import print_function
from __future__ import absolute_import
# ============================================================================
# Copyright 2017 BRAIN Corporation. All rights reserved. This software is
# provided to you under BRAIN Corporation's Beta License Agreement and
# your use of the software is governed by the terms of that Beta License
# Agreement, found at http://www.braincorporation.com/betalicense.
# ============================================================================

import numpy as np
import pytest

from bc_gym_planning_env.utilities.frozenarray import freeze_array


def test_frozen_array():
    a = freeze_array(np.array([0, 1, 2]))
    np.testing.assert_array_equal(a, [0, 1, 2])

    assert a[0] == 0

    with pytest.raises(ValueError):
        a[0] = 3

    b = a[:1]
    assert len(b) == 1
    assert b[0] == 0
    with pytest.raises(ValueError):
        b[0] = 4

    # some slicing operations create a copy that is mutable again,
    # however original array is still immutable
    c = a[[0, 2]]
    np.testing.assert_array_equal(c, [0, 2])
    c[1] = 10
    np.testing.assert_array_equal(a, [0, 1, 2])
    with pytest.raises(ValueError):
        a[0] = 3


if __name__ == '__main__':
    test_frozen_array()
