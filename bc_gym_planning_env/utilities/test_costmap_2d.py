from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from bc_gym_planning_env.utilities.costmap_2d import CostMap2D


def test_world_size_and_center():
    costmap = CostMap2D.create_empty((1, 1), 0.5, world_origin=(0, 0))
    assert costmap.get_data().shape == (2, 2)
    np.testing.assert_array_equal(costmap.world_bounds(), [0, 1, 0, 1])
    np.testing.assert_array_equal(costmap.world_size(), [1, 1])
    np.testing.assert_array_equal(costmap.world_center(), [0.5, 0.5])

    costmap = CostMap2D.create_empty((1, 1), 0.5, world_origin=(-0.13, 0.23))
    assert costmap.get_data().shape == (2, 2)
    np.testing.assert_array_equal(costmap.world_bounds(), [-0.13, 1-0.13, 0.23, 1+0.23])
    np.testing.assert_array_equal(costmap.world_size(), [1, 1])
    np.testing.assert_array_equal(costmap.world_center(), [0.5-0.13, 0.5+0.23])

    costmap = CostMap2D.create_empty((1, 1), 0.05, world_origin=(-0.13, 0.23))
    assert costmap.get_data().shape == (20, 20)
    np.testing.assert_array_equal(costmap.world_bounds(), [-0.13, 1-0.13, 0.23, 1+0.23])
    np.testing.assert_array_equal(costmap.world_size(), [1, 1])
    np.testing.assert_array_equal(costmap.world_center(), [0.5-0.13, 0.5+0.23])

    costmap = CostMap2D.create_empty((1, 1), 0.03, world_origin=(-0.13, 0.23))
    assert costmap.get_data().shape == (33, 33)
    # if there is not an exact match with resolution, we create smaller map than requested
    np.testing.assert_array_equal(costmap.world_bounds(), [-0.13, 1-0.13-0.01, 0.23, 1+0.23-0.01])
    np.testing.assert_array_equal(costmap.world_size(), [1-0.01, 1-0.01])
    np.testing.assert_array_equal(costmap.world_center(), [0.5-0.13-0.005, 0.5+0.23-0.005])


if __name__ == '__main__':
    test_world_size_and_center()
