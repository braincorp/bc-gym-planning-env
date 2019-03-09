from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.utilities.map_drawing_utils import get_drawing_coordinates_from_physical, \
    get_physical_coords_from_drawing, draw_trajectory, draw_robot, prepare_canvas


def _tricycle_footprint():
    """Realistic tricycle footprint for testing"""
    return np.array([
        [1348.35, 0.],
        [1338.56, 139.75],
        [1306.71, 280.12],
        [1224.36, 338.62],
        [1093.81, 374.64],
        [-214.37, 374.64],
        [-313.62, 308.56],
        [-366.36, 117.44],
        [-374.01, -135.75],
        [-227.96, -459.13],
        [-156.72, -458.78],
        [759.8, -442.96],
        [849.69, -426.4],
        [1171.05, -353.74],
        [1303.15, -286.54],
        [1341.34, -118.37]
    ]) / 1000.


def test_phys_to_drawing_coordinates_conversion():
    np.testing.assert_array_equal(
        get_drawing_coordinates_from_physical((1, 1), 1., np.array((0., 0)), np.array((0., 0))),
        (0, 0))

    np.testing.assert_array_equal(
        get_drawing_coordinates_from_physical((2, 2), 1., np.array((0., 0)), np.array((0., 0))),
        (0, 1))

    np.testing.assert_array_equal(
        get_drawing_coordinates_from_physical((2, 2), 1., np.array((0., 0)), np.array((1., 1))),
        (1, 0))

    np.testing.assert_array_equal(
        get_drawing_coordinates_from_physical((2, 2), 1., np.array((0., 0)), np.array((0., 1))),
        (0, 0))

    np.testing.assert_array_equal(
        get_drawing_coordinates_from_physical((10, 10), 1., np.array((0., 0)), np.array((0., 0))),
        (0, 9))

    np.testing.assert_array_equal(
        get_drawing_coordinates_from_physical((10, 10), 1., np.array((0., 0)), np.array((2., 3))),
        (2, 6))

    np.testing.assert_array_equal(
        get_drawing_coordinates_from_physical((10, 10), 0.5, np.array((0., 0)), np.array((2., 3))),
        (4, 3))

    # vectorized
    np.testing.assert_array_equal(
        get_drawing_coordinates_from_physical((10, 10), 0.5, np.array((0., 0)), np.array([[0., 0], [1, 1], [2, 3]])),
        [[0, 9], [2, 7], [4, 3]])

    # map shape is 3d
    np.testing.assert_array_equal(
        get_drawing_coordinates_from_physical((10, 10, 3), 0.5, np.array((0., 0)), np.array((2., 3))),
        (4, 3))


def test_drawing_to_phys_coordinates_conversion():
    np.testing.assert_array_equal(
        get_physical_coords_from_drawing((1, 1), 1., (0, 0), (0, 0)),
        (0, 0))

    drawing = np.array([[0, 9], [2, 7], [4, 3]])
    np.testing.assert_array_equal(
        get_physical_coords_from_drawing((10, 10), 0.5, (0, 0), drawing),
        [[0, 0], [1, 1], [2, 3]])
    # check that coordinates are not corrupted by the transformation
    np.testing.assert_array_equal(
        drawing,
        [[0, 9], [2, 7], [4, 3]])

    # map shape is 3d
    np.testing.assert_array_equal(
        get_physical_coords_from_drawing((1, 1, 3), 1., (0, 0), (0, 0)),
        (0, 0))


def test_draw_trajectory():
    costmap = CostMap2D(
        data=np.zeros((400, 400), dtype=np.uint8),
        origin=np.array([-10., -10.]),
        resolution=0.05
    )
    trajectory_picture = np.zeros((costmap.get_data().shape + (3,)), dtype=np.uint8)

    trajectory = np.array([[-5, -2, np.pi/4],
                           [-4.5, -1.5, np.pi/3]])
    draw_trajectory(
        trajectory_picture, costmap.get_resolution(), costmap.get_origin(),
        trajectory, color=(0, 255, 0))

    idy, idx = np.where(np.all(trajectory_picture == (0, 255, 0), axis=2))
    np.testing.assert_array_equal(
        idy, [229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239])
    np.testing.assert_array_equal(
        idx, [110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100])

    draw_trajectory(trajectory_picture, costmap.get_resolution(), costmap.get_origin(), [])


def test_draw_robot_full():
    costmap = CostMap2D.create_empty((5, 5), 0.05)
    array_to_draw = np.zeros((costmap.get_data().shape + (3,)), dtype=np.uint8)
    draw_robot(array_to_draw,
               _tricycle_footprint(),
               (2, 2, np.pi/4),
               costmap.get_resolution(),
               costmap.get_origin(),
               color=(0, 0, 150))

    draw_robot(array_to_draw,
               _tricycle_footprint(),
               (-0.5, -0.5, np.pi/4),
               costmap.get_resolution(),
               costmap.get_origin(),
               color=(0, 0, 150))

    draw_robot(array_to_draw,
               _tricycle_footprint(),
               (4.6, 3, np.pi/4),
               costmap.get_resolution(),
               costmap.get_origin(),
               color=(0, 0, 150))

    draw_robot(array_to_draw,
               _tricycle_footprint(),
               (-4234.6, 3456, np.pi/4),
               costmap.get_resolution(),
               costmap.get_origin(),
               color=(0, 0, 150))

    static_map = CostMap2D.create_empty((5, 5), 0.05)

    img = prepare_canvas(static_map.get_data().shape)
    draw_robot(
        img,
        _tricycle_footprint(),
        [0., -3, np.pi / 2.],
        static_map.get_resolution(),
        static_map.get_origin())


if __name__ == "__main__":
    test_phys_to_drawing_coordinates_conversion()
    test_drawing_to_phys_coordinates_conversion()
    test_draw_trajectory()
    test_draw_robot_full()
