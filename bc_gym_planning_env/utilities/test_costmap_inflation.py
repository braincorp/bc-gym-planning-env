from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.utilities.costmap_inflation import inflate_costmap
from bc_gym_planning_env.utilities.map_drawing_utils import add_wall_to_static_map
from bc_gym_planning_env.utilities.costmap_inflation import INSCRIBED_INFLATED_OBSTACLE


def _rectangular_footprint():
    """Realistic rectangular footprint for testing"""
    return np.array([
        [-0.77, -0.385],
        [-0.77, 0.385],
        [0.67, 0.385],
        [0.67, -0.385]])


def test_costmap_inflation():
    """Sanity checks on inflation"""
    costmap = CostMap2D.create_empty((1, 1), 0.1, (0, 0))
    add_wall_to_static_map(costmap, (0, 0), (1, 1))

    cost_scaling_factor = 1.
    inflated_costmap = inflate_costmap(costmap, cost_scaling_factor, _rectangular_footprint())

    inflated_pixels = len(np.where(inflated_costmap.get_data() == INSCRIBED_INFLATED_OBSTACLE)[0])
    assert inflated_pixels == 70


if __name__ == '__main__':
    test_costmap_inflation()
