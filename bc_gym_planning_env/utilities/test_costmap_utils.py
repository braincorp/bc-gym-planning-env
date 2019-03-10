from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from bc_gym_planning_env.utilities.coordinate_transformations import world_to_pixel

from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.utilities.costmap_utils import extract_egocentric_costmap, rotate_costmap, \
    is_robot_colliding, in_costmap_bounds
from bc_gym_planning_env.utilities.map_drawing_utils import add_wall_to_static_map
from bc_gym_planning_env.utilities.path_tools import get_pixel_footprint, get_blit_values


def assert_mark_at(costmap2d, x, y, check_inflation=False, check_single=True):
    # a particular example of robot's inscribed radius for testing
    robot_inscribed_radius = 0.385
    pixel_inscribed_radius = robot_inscribed_radius / costmap2d.get_resolution()
    cost = costmap2d.get_data()
    lethal = np.where(cost == 254)
    inscribed = np.where(cost == 253)
    non_inscribed = np.where(cost < 253)
    d_inscribed = np.hypot(inscribed[0] - y, inscribed[1] - x)
    d_non_inscribed = np.hypot(non_inscribed[0] - y, non_inscribed[1] - x)
    if check_single:
        if len(lethal[0]) != 1:
            raise AssertionError("Expected to see a signle mark,but got at %s" % (lethal,))
    if cost[int(y), int(x)] != 254:
        raise AssertionError("Expected to see a mark at %s but see at %s" % ((x, y), (lethal[1][0], lethal[0][0])))
    if check_inflation:
        assert len(inscribed[0]) > 0.95 * np.pi * pixel_inscribed_radius ** 2
        assert len(inscribed[0]) < 1.05 * np.pi * pixel_inscribed_radius ** 2
        assert np.all(d_inscribed >= 1.)
        assert np.all(d_inscribed <= pixel_inscribed_radius)
        assert np.all(d_non_inscribed > pixel_inscribed_radius)


def test_costmap_2d_extract_egocentric_costmap():
    costmap2d = CostMap2D(np.zeros((100, 100), dtype=np.float64), 0.05, np.array([0.0, 0.0]))
    costmap2d.get_data()[10, 20] = CostMap2D.LETHAL_OBSTACLE

    np.testing.assert_array_equal(costmap2d.world_bounds(), (0, 5, 0, 5))
    assert_mark_at(costmap2d, 20, 10)

    # # dummy cut
    cut_map = extract_egocentric_costmap(costmap2d, (0., 0., 0.))
    assert cut_map.get_resolution() == costmap2d.get_resolution()
    np.testing.assert_array_equal(cut_map.get_origin(), (0.0, 0.0))
    np.testing.assert_array_equal(cut_map.get_data().shape, (100, 100))
    np.testing.assert_array_equal(cut_map.world_bounds(), (0.0, 5, 0.0, 5))
    assert_mark_at(cut_map, 20, 10)

    # shift
    cut_map = extract_egocentric_costmap(costmap2d, (0.2, 0.2, 0.0))
    assert cut_map.get_resolution() == costmap2d.get_resolution()
    np.testing.assert_array_equal(cut_map.get_origin(), (-0.2, -0.2))
    np.testing.assert_array_equal(cut_map.get_data().shape, (100, 100))
    np.testing.assert_array_equal(cut_map.world_bounds(), (-0.2, 4.8, -0.2, 4.8))

    # ego centric view doesn't shift the data itself if not rotated
    assert_mark_at(cut_map, 20, 10)

    # rotate so that mark is almost in the front of the robot
    cut_map = extract_egocentric_costmap(costmap2d, (0.0, 0.0, np.pi/6-0.05))
    assert cut_map.get_resolution() == costmap2d.get_resolution()
    np.testing.assert_array_equal(cut_map.get_origin(), (-0., -0.))
    np.testing.assert_array_equal(cut_map.get_data().shape, (100, 100))
    np.testing.assert_array_equal(cut_map.world_bounds(), (0.0, 5, 0.0, 5))

    assert_mark_at(cut_map, 22, 0, check_single=False)

    # rotate as if robot is in the center of the costmap
    cut_map = extract_egocentric_costmap(costmap2d, (2.5, 2.5, -np.pi / 2.))
    assert cut_map.get_resolution() == costmap2d.get_resolution()
    np.testing.assert_array_equal(cut_map.get_origin(), (-2.5, -2.5))
    np.testing.assert_array_equal(cut_map.get_data().shape, (100, 100))
    np.testing.assert_array_equal(cut_map.world_bounds(), (-2.5, 2.5, -2.5, 2.5))
    assert_mark_at(cut_map, 90, 20)

    # rotate as if robot is in the center of the costmap with explicit parameters
    cut_map = extract_egocentric_costmap(
        costmap2d,
        (2.5, 2.5, -np.pi/2),
        resulting_origin=np.array([-2.5, -2.5], dtype=np.float64),
        resulting_size=np.array((5., 5.)))
    assert cut_map.get_resolution() == costmap2d.get_resolution()
    np.testing.assert_array_equal(cut_map.get_origin(), (-2.5, -2.5))
    np.testing.assert_array_equal(cut_map.get_data().shape, (100, 100))
    np.testing.assert_array_equal(cut_map.world_bounds(), (-2.5, 2.5, -2.5, 2.5))
    assert_mark_at(cut_map, 90, 20)

    # rotate as if robot is in the center of the costmap with smaller size
    cut_map = extract_egocentric_costmap(
        costmap2d,
        (2.5, 2.5, -np.pi/2),
        resulting_origin=np.array([-2.5, -2.5], dtype=np.float64),
        resulting_size=(4.6, 4.9))
    assert cut_map.get_resolution() == costmap2d.get_resolution()
    np.testing.assert_array_equal(cut_map.get_origin(), (-2.5, -2.5))
    np.testing.assert_array_equal(cut_map.get_data().shape, (98, 92))
    np.testing.assert_array_almost_equal(cut_map.world_bounds(), (-2.5, 2.1, -2.5, 2.4))
    assert_mark_at(cut_map, 90, 20)

    # do not rotate but shift
    cut_map = extract_egocentric_costmap(
        costmap2d,
        (2.5, 2.5, 0.0),
        resulting_origin=np.array([-5, -4], dtype=np.float64),
        resulting_size=(5, 5))
    assert cut_map.get_resolution() == costmap2d.get_resolution()
    np.testing.assert_array_equal(cut_map.get_origin(), (-5, -4))
    np.testing.assert_array_equal(cut_map.get_data().shape, (100, 100))
    np.testing.assert_array_almost_equal(cut_map.world_bounds(), (-5, 0, -4, 1))
    assert_mark_at(cut_map, 70, 40)

    # do not rotate but shift and cut
    cut_map = extract_egocentric_costmap(
        costmap2d,
        (2.5, 2.5, 0.0),
        resulting_origin=np.array([-5, -4], dtype=np.float64),
        resulting_size=(4, 4))
    assert cut_map.get_resolution() == costmap2d.get_resolution()
    np.testing.assert_array_equal(cut_map.get_origin(), (-5, -4))
    np.testing.assert_array_equal(cut_map.get_data().shape, (80, 80))
    np.testing.assert_array_almost_equal(cut_map.world_bounds(), (-5, -1, -4, 0))
    assert_mark_at(cut_map, 70, 40)

    # rotate, shift and cut -> extract ego costmap centered on the robot
    cut_map = extract_egocentric_costmap(
        costmap2d,
        (1.5, 1.5, -np.pi/4),
        resulting_origin=np.array([-2, -2], dtype=np.float64),
        resulting_size=(4, 4))
    assert cut_map.get_resolution() == costmap2d.get_resolution()
    np.testing.assert_array_equal(cut_map.get_origin(), (-2, -2))
    np.testing.assert_array_equal(cut_map.get_data().shape, (80, 80))
    np.testing.assert_array_almost_equal(cut_map.world_bounds(), (-2, 2, -2, 2))
    assert_mark_at(cut_map, 47, 19)

    # rotate, shift and expand -> extract ego costmap centered on the robot, going off the limits of original map
    cut_map = extract_egocentric_costmap(
        costmap2d,
        (1., 1.5, -np.pi/4),
        resulting_origin=np.array([-3, -3], dtype=np.float64),
        resulting_size=(7, 6)
    )
    assert cut_map.get_resolution() == costmap2d.get_resolution()
    np.testing.assert_array_equal(cut_map.get_origin(), (-3, -3))
    np.testing.assert_array_equal(cut_map.get_data().shape, (120, 140))
    np.testing.assert_array_almost_equal(cut_map.world_bounds(), (-3, 4, -3, 3))
    assert_mark_at(cut_map, 74, 46)

    costmap_with_image = CostMap2D(np.zeros((100, 100, 3), dtype=np.float64), 0.05, np.array([0., 0.]))
    extract_egocentric_costmap(costmap_with_image, [0., 0., np.pi/4])


def test_costmap_2d_extract_egocentric_costmap_with_nonzero_origin():
    costmap2d = CostMap2D(np.zeros((100, 100), dtype=np.float64), 0.05, np.array([1.0, 2.0]))
    costmap2d.get_data()[10, 20] = CostMap2D.LETHAL_OBSTACLE

    # rotate, shift and cut -> extract ego costmap centered on the robot
    cut_map = extract_egocentric_costmap(
        costmap2d,
        (3.5, 3.5, -np.pi/4),
        resulting_origin=np.array([-2, -2], dtype=np.float64),
        resulting_size=(4, 4))
    assert cut_map.get_resolution() == costmap2d.get_resolution()
    np.testing.assert_array_equal(cut_map.get_origin(), (-2, -2))
    np.testing.assert_array_equal(cut_map.get_data().shape, (80, 80))
    np.testing.assert_array_almost_equal(cut_map.world_bounds(), (-2, 2, -2, 2))
    assert_mark_at(cut_map, 33, 5)


def test_costmap_2d_extract_egocentric_binary_costmap():
    costmap = CostMap2D.create_empty((5, 5), 0.05)
    costmap.get_data()[30:32, 30:32] = CostMap2D.LETHAL_OBSTACLE

    cut_map = extract_egocentric_costmap(
        costmap,
        (1.5, 1.5, -np.pi/4),
        resulting_origin=np.array([-2.0, -2.0], dtype=np.float64),
        resulting_size=(4., 4))

    rotated_data = cut_map.get_data()
    non_free_indices = np.where(rotated_data != 0)
    np.testing.assert_array_equal(non_free_indices[0], [40, 41, 41, 41, 42])
    np.testing.assert_array_equal(non_free_indices[1], [40, 39, 40, 41, 40])

    assert (rotated_data[non_free_indices[0], non_free_indices[1]] == CostMap2D.LETHAL_OBSTACLE).all()


def test_rotate_costmap():
    # test inflated map
    costmap = CostMap2D.create_empty((5, 4), 0.05)
    costmap.get_data()[30, 30] = CostMap2D.LETHAL_OBSTACLE

    assert_mark_at(costmap, 30, 30)
    rotated_costmap = CostMap2D(rotate_costmap(costmap.get_data(), -np.pi/4),
                                costmap.get_resolution(), costmap.get_origin())
    assert_mark_at(rotated_costmap, 29, 47, check_inflation=False)

    # test binary map
    costmap = CostMap2D.create_empty((5, 4), 0.05)
    costmap.get_data()[30, 30] = CostMap2D.LETHAL_OBSTACLE

    rotated_data = rotate_costmap(costmap.get_data(), -np.pi/4)

    non_free_indices = np.where(rotated_data != 0)
    assert len(non_free_indices[0]) == 1
    non_free_indices = (non_free_indices[0][0], non_free_indices[1][0])
    assert non_free_indices == (47, 29)
    assert(rotated_data[non_free_indices[0], non_free_indices[1]] == CostMap2D.LETHAL_OBSTACLE)


def is_robot_colliding_reference(robot_pose, footprint, costmap_data, origin, resolution):
    """
    Pure python implementation of is_robot_colliding
    Check costmap for obstacle at (world_x, world_y)
    If orientation is None, obstacle detection will check only the inscribed-radius
    distance collision. This means that, if the robot is not circular,
    there may be an undetected orientation-dependent collision.
    If orientation is given, footprint must also be given, and should be the same
    used by the costmap to inflate costs. Proper collision detection will
    then be done.
    """
    assert isinstance(robot_pose, np.ndarray)
    # convert to costmap coordinate system:
    map_x, map_y = world_to_pixel(robot_pose[:2], origin, resolution)
    # TODO: remove this because the following is technically wrong: even if robot's origin is outside, it still can collide
    if not in_costmap_bounds(costmap_data, map_x, map_y):
        return False
    cost = costmap_data[map_y, map_x]
    if cost in [CostMap2D.LETHAL_OBSTACLE]:
        return True

    # Now check for orientation-dependent collision
    fp = get_pixel_footprint(robot_pose[2], footprint, resolution)
    values = get_blit_values(fp, costmap_data, map_x, map_y)
    return (values == CostMap2D.LETHAL_OBSTACLE).any()


def _rectangular_footprint():
    """Realistic rectangular footprint for testing"""
    return np.array([
        [-0.77, -0.385],
        [-0.77, 0.385],
        [0.67, 0.385],
        [0.67, -0.385]])


def test_is_robot_colliding():
    costmap = CostMap2D.create_empty((10, 6), 0.05, (-1, -3))
    wall_pos_x = 3.9
    add_wall_to_static_map(costmap, (wall_pos_x, -4.), (wall_pos_x, -1+1.5))
    wall_pos_x = 1.5
    add_wall_to_static_map(costmap, (wall_pos_x, -4.), (wall_pos_x, -1+1.5))
    add_wall_to_static_map(costmap, (5., -4.), (5. + 1, -1 + 4.5))
    footprint = _rectangular_footprint()

    poses = np.array([
        (0., 0., 0.2),
        (1., 0., 0.2),
        (2., 0., 0.2),
        (3., 0., 0.2),
        (4., 0., 0.2),
        (5., 0., 0.2),
        (6., 0., 0.2),
        (0., 1.2, np.pi/2+0.4),
        (1., 1.2, np.pi/2+0.4),
        (2., 1.2, np.pi/2+0.4),
        (3., 1.2, np.pi/2+0.4),
        (4., 1.2, np.pi/2+0.4),
        (5., 1.2, np.pi/2+0.4),
        (6., 1.2, np.pi/2+0.4),
        (0., -3, 0.2),
        (1., -3, 0.2),
        (2., -3, 0.2),
        # These cases should be colliding but because of the current bug,
        # they are not because robot's center is off bounds and we do not want to change behavior at this point
        (0., -3.2, 0.2),
        (1., -3.2, 0.2),
        (2., -3.2, 0.2),
    ])

    expected_collisions = [
        False,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        False,
        False,
        False,
    ]

    assert len(poses) == len(expected_collisions)

    for robot_pose, expected in zip(poses, expected_collisions):
        is_colliding = is_robot_colliding(
            robot_pose, footprint, costmap.get_data(), costmap.get_origin(), costmap.get_resolution()
        )
        assert is_colliding == expected

    random_poses = np.random.rand(1000, 3)
    random_poses[:, :2] *= costmap.world_size() + costmap.get_origin() + np.array([1., 1.])
    for pose in random_poses:
        is_colliding = is_robot_colliding(
            pose, footprint, costmap.get_data(), costmap.get_origin(), costmap.get_resolution()
        )
        is_colliding_ref = is_robot_colliding_reference(
            pose, footprint, costmap.get_data(), costmap.get_origin(), costmap.get_resolution()
        )
        assert is_colliding == is_colliding_ref


if __name__ == '__main__':
    test_rotate_costmap()
    test_costmap_2d_extract_egocentric_costmap()
    test_costmap_2d_extract_egocentric_costmap_with_nonzero_origin()
    test_costmap_2d_extract_egocentric_binary_costmap()
    test_is_robot_colliding()
