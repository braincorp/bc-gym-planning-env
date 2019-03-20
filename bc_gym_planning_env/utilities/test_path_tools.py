from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import pytest
import cv2

from bc_gym_planning_env.utilities.path_tools import blit, get_blit_values, ensure_float_numpy, orient_path, \
    path_velocity, pose_distances, distance_to_segments, distances_to_segment, distances_to_multiple_segments, \
    inscribed_radius, circumscribed_radius, get_pixel_footprint, compute_robot_area


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


def _rectangular_footprint():
    """Realistic rectangular footprint for testing"""
    return np.array([
        [-0.77, -0.385],
        [-0.77, 0.385],
        [0.67, 0.385],
        [0.67, -0.385]])


def test_blit():
    '''
    Test blitting corner cases - blit half outside costmap, fully outside, etc
    '''
    for depth in (1, 3):
        if depth == 1:
            im = np.zeros((20, 50), dtype=np.uint8)
        else:
            im = np.zeros((20, 50, 3), dtype=np.uint8)
        patch = np.zeros((6, 4), dtype=np.uint8)
        patch[2:4, :] = 255
        for ((x, y), xlim, ylim) in [((10, 10), (8, 12), (9, 11)),
                                     ((0, 10), (0, 2), (9, 11)),
                                     ((50, 10), (48, 50), (9, 11)),
                                     ((10, 0), (8, 12), (0, 1)),
                                     ((10, 20), (8, 12), (19, 20)),
                                     ((0, 0), (0, 2), (0, 1)),
                                     ((50, 20), (48, 50), (19, 20)),
                                     ((0, 20), (0, 2), (19, 20)),
                                     ((50, 0), (48, 50), (0, 1)),
                                     ((-1, 20), (0, 1), (19, 20)),
                                     ((51, 0), (49, 50), (0, 1)),
                                     ((10, -1), None, None),
                                     ((10, 21), None, None),
                                     ((-2, 10), None, None),
                                     ((52, 10), None, None),
                                     ((-100, 10), None, None),
                                     ((200, 10), None, None),
                                     ((200, 200), None, None),
                                     ((10, -100), None, None),
                                     ((-5, -5), None, None)] +\
                                    [((x, y), None, None) for y in range(22, 50) for x in [-10, 60, 3]] + \
                                    [((x, y), None, None) for y in range(-30, 0) for x in [-10, 60, 3]] + \
                                    [((x, y), None, None) for x in range(52, 100) for y in [-10, 30, 2]] + \
                                    [((x, y), None, None) for x in range(-30, -1) for y in [-10, 30, 2]]:
            blit(patch, im, x, y, 100)
            if xlim is None or ylim is None:
                assert np.all(im == 0)
                assert np.sum(get_blit_values(patch, im, x, y)) == 0
            else:
                assert np.all(im[ylim[0]:ylim[1], xlim[0]:xlim[1], ...] == 100)
                assert np.sum(im) == depth * 100 * (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])
                assert np.sum(get_blit_values(patch, im, x, y)) == np.sum(im)
            im[:] = 0

    # Finally test a blit bigger than the image
    im = np.zeros((3, 3), dtype=np.uint8)
    patch = np.zeros((5, 5), dtype=np.uint8)
    patch[2:4, :] = 255
    x, y = 1, 1
    xlim = (0, 3)
    ylim = (1, 3)
    blit(patch, im, x, y, 100)
    assert np.all(im[ylim[0]:ylim[1], xlim[0]:xlim[1], ...] == 100)
    assert np.sum(im) == 100 * (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])
    assert np.sum(get_blit_values(patch, im, x, y)) == np.sum(im)


def test_ensure_float_numpy():
    assert ensure_float_numpy([1]).dtype == np.float
    assert ensure_float_numpy([1.]).dtype == np.float
    assert ensure_float_numpy(np.array([1.])).dtype == np.float
    assert ensure_float_numpy(np.array([1.], dtype=float)).dtype == np.float
    assert ensure_float_numpy(np.array([1.], dtype=np.float64)).dtype == np.float
    assert ensure_float_numpy(np.array([1.], dtype=np.float32)).dtype == np.float32
    with pytest.raises(AssertionError):
        ensure_float_numpy(np.array([1]))

    assert ensure_float_numpy(np.array([[1., 2], [3, 4]])).shape == (2, 2)


def test_orient_path():
    path = np.array([[0., 0.],
                     [0., 1.]])
    np.testing.assert_array_almost_equal(
        orient_path(path),
        [[0., 0., np.pi / 2.],
         [0., 1., np.pi / 2.]]
    )

    # int path was not handled correctly before
    path = np.array([[0., 0.],
                     [0., 1.]], dtype=int)
    with pytest.raises(AssertionError):
        orient_path(path)

    path = np.array([[0., 0.],
                     [1., 1.]])
    np.testing.assert_array_almost_equal(
        orient_path(path),
        [[0., 0., np.pi/4],
         [1., 1., np.pi/4]]
    )

    path = np.array([[0., 0.],
                     [1., 1.],
                     [0., 0.]])
    np.testing.assert_array_almost_equal(
        orient_path(path),
        [[0., 0., np.pi/4],
         [1., 1., -3*np.pi/4],
         [0., 0., -3*np.pi/4]]
    )

    path = np.array([[0.0, 0.0]])
    robot_pose = np.concatenate((path[0], [0.0]), axis=0)
    np.testing.assert_array_almost_equal(
        orient_path(path, robot_pose=robot_pose, final_pose=robot_pose,
                    max_movement_for_turn_in_place=0.01),
        np.array([robot_pose])
    )

    path = np.array([[0.0, 0.0],
                     [0.0, 1.0]])
    robot_pose = np.concatenate((path[0], [0.0]), axis=0)
    final_pose = np.concatenate((path[-1], [0.0]), axis=0)
    np.testing.assert_array_almost_equal(
        orient_path(path[:, :2], robot_pose=robot_pose, final_pose=final_pose,
                    max_movement_for_turn_in_place=0.01),
        [[0.0, 0.0, 0.5 * np.pi],
         [0.0, 1.0, 0.0]]
    )

    path = np.array([[0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0]])
    robot_pose = np.concatenate((path[0], [0.0]), axis=0)
    final_pose = np.concatenate((path[-1], [np.pi]), axis=0)
    np.testing.assert_array_almost_equal(
        orient_path(path[:, :2], robot_pose=robot_pose, final_pose=final_pose,
                    max_movement_for_turn_in_place=0.01),
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.5 * np.pi],
         [0.0, 0.0, np.pi]]
    )

    path = np.array([[0.0, 0.0],
                     [0.0005, -0.0005],
                     [-0.0005, -0.0005],
                     [-0.0005, 0.0005],
                     [0.0, 0.0],
                     [0.0, 0.1],
                     [0.0, 0.3]])
    robot_pose = np.concatenate((path[0], [0.0]), axis=0)
    final_pose = np.concatenate((path[-1], [np.pi * 0.5]), axis=0)
    np.testing.assert_array_almost_equal(
        orient_path(path[:, :2], robot_pose=robot_pose, final_pose=final_pose,
                    max_movement_for_turn_in_place=0.01),
        [[0.0, 0.0, 0.0],
         [0.0005, -0.0005, np.pi * 0.125],
         [-0.0005, -0.0005, np.pi * 0.25],
         [-0.0005, 0.0005, np.pi * 0.375],
         [0.0, 0.0, np.pi * 0.5],
         [0.0, 0.1, np.pi * 0.5],
         [0.0, 0.3, np.pi * 0.5]]
    )

    # Test input path length == 1
    np.testing.assert_array_almost_equal(
        orient_path(np.array([[0., 0.]])),
        [[0., 0., 0.]]
    )

    # Test input path length == 0
    np.testing.assert_array_equal(
        orient_path(np.array([])),
        np.empty(shape=(0, 3))
    )


def test_path_velocity():
    v, w = path_velocity(np.array((
        (0., 0, 0, 0),
        (1., 0, 0, 0),
    )))
    np.testing.assert_array_almost_equal(v, (0,))
    np.testing.assert_array_almost_equal(w, (0,))

    v, w = path_velocity(np.array((
        (0., 0, 0, 0),
        (1., 1, 0, 0),
    )))
    np.testing.assert_array_almost_equal(v, (1,))
    np.testing.assert_array_almost_equal(w, (0,))

    v, w = path_velocity(np.array((
        (0., 0, 0, 1),
        (1., 0, 1, 1),
    )))
    np.testing.assert_array_almost_equal(v, (1,))
    np.testing.assert_array_almost_equal(w, (0,))

    v, w = path_velocity(np.array((
        (0., 0, 0, 0),
        (1., 0, 0, 1),
    )))
    np.testing.assert_array_almost_equal(v, (0,))
    np.testing.assert_array_almost_equal(w, (1,))

    v, w = path_velocity(np.array((
        (0., 0, 0, 0),
        (1., 3, 4, 1),
    )))
    np.testing.assert_array_almost_equal(v, (5,))
    np.testing.assert_array_almost_equal(w, (1,))

    v, w = path_velocity(np.array((
        (0, 0, 0, 0),
        (0.1, 3, 4, 1),
        (0.2, 3, 5, -1),
    )))
    np.testing.assert_array_almost_equal(v, (50, 10))
    np.testing.assert_array_almost_equal(w, (10, -20))

    v, w = path_velocity(np.array((
        (0, 0, 0, 0),
        (0.1, -3, -4, 1),
        (0.2, 0, 0, 0),
    )))
    np.testing.assert_array_almost_equal(v, (-50, 50))
    np.testing.assert_array_almost_equal(w, (10, -10))

    v, w = path_velocity(np.array((
        (0, 0, 0, np.pi / 2.-0.01),
        (1, 0, -0.1, np.pi / 2.-0.01),
    )))
    np.testing.assert_array_almost_equal(v, (-0.1))
    np.testing.assert_array_almost_equal(w, (0,))


def test_pose_distances():
    np.testing.assert_almost_equal(
        pose_distances(np.array([0., 0., 0.]), np.array([1., 1., 1])), (np.sqrt(2), 1)
    )

    np.testing.assert_almost_equal(
        pose_distances(np.array([-1., -1., -1]), np.array([1., 1., 1])), (2*np.sqrt(2), 2)
    )

    np.testing.assert_almost_equal(
        pose_distances(np.array([-1., -1., -np.pi+0.1]), np.array([1., 1., np.pi-0.1])), (2*np.sqrt(2), 0.2)
    )

    np.testing.assert_almost_equal(
        pose_distances(np.array([[0., 0., 0.], [-1., -1., -np.pi+0.1]]),
                       np.array([[1., 1., 1], [1., 1., np.pi-0.1]])),
        [[np.sqrt(2), 2*np.sqrt(2)],
         [1, 0.2]]
    )


def test_distance_to_segments():
    for point in [[0., 0.], [-2., -3.], [1., 0.]]:
        p = np.array(point)
        # normal is on the segment
        assert abs(distance_to_segments(p, p + [[1, -1]], p + [[1, 1]]) - 1.) < 1e-6
        assert abs(distance_to_segments(p, p + [[-1, 1]], p + [[1, 1]]) - 1.) < 1e-6
        assert abs(distance_to_segments(p, p + [[-1, -1]], p + [[1, -1]]) - 1.) < 1e-6
        assert abs(distance_to_segments(p, p + [[-1, -1]], p + [[-1, 1]]) - 1.) < 1e-6
        assert abs(distance_to_segments(p, p + [[0, -1]], p + [[0, 1]]) - 0) < 1e-6
        assert abs(distance_to_segments(p, p + [[1, 0]], p + [[0, 1]]) - np.sqrt(2)/2) < 1e-6

        # normal is not on the segment
        assert abs(distance_to_segments(p, p + [[1, 1]], p + [[1, 2]]) - np.sqrt(2)) < 1e-6
        assert abs(distance_to_segments(p, p + [[1, 2]], p + [[1, 1]]) - np.sqrt(2)) < 1e-6
        assert abs(distance_to_segments(p, p + [[0, 1]], p + [[0, 2]]) - 1) < 1e-6
        assert abs(distance_to_segments(p, p + [[1, 0]], p + [[2, 0]]) - 1) < 1e-6


def test_distances_to_segment():
    for point in [[0., 0.], [-2., -3.], [1., 0.]]:
        p = np.array(point)  # 2-vector
        pp = p[None, :]  # 1 x 2 matrix
        # normal is on the segment
        assert abs(distances_to_segment(pp, p + [1, -1], p + [1, 1]) - 1.) < 1e-6
        assert abs(distances_to_segment(pp, p + [-1, 1], p + [1, 1]) - 1.) < 1e-6
        assert abs(distances_to_segment(pp, p + [-1, -1], p + [1, -1]) - 1.) < 1e-6
        assert abs(distances_to_segment(pp, p + [-1, -1], p + [-1, 1]) - 1.) < 1e-6
        assert abs(distances_to_segment(pp, p + [0, -1], p + [0, 1]) - 0) < 1e-6
        assert abs(distances_to_segment(pp, p + [1, 0], p + [0, 1]) - np.sqrt(2)/2) < 1e-6

        # normal is not on the segment
        assert abs(distances_to_segment(pp, p + [1, 1], p + [1, 2]) - np.sqrt(2)) < 1e-6
        assert abs(distances_to_segment(pp, p + [1, 2], p + [1, 1]) - np.sqrt(2)) < 1e-6
        assert abs(distances_to_segment(pp, p + [0, 1], p + [0, 2]) - 1) < 1e-6
        assert abs(distances_to_segment(pp, p + [1, 0], p + [2, 0]) - 1) < 1e-6

        # test several points at a time
        assert(np.all(np.abs(distances_to_segment(np.vstack((pp, pp + [1., 0], pp + [0., 1.], pp + [1.5, 0.8])),
                                                  p + [1, 0], p + [2, 0]) - [1, 0., np.sqrt(2), 0.8]) < 1e-6))


def test_distances_to_multiple_segments():
    p = np.array([-2., -3.])
    distances = distances_to_multiple_segments(
        p,
        segment_origins=np.array(
            [p + [1, -1],
             p + [-1, 1],
             p + [-1, -1],
             p + [-1, -1],
             p + [0, -1],
             p + [1, 0],
             p + [1, 1],
             p + [1, 2],
             p + [0, 1],
             p + [1, 0]
             ]),
        segment_ends=np.array(
            [p + [1, 1],
             p + [1, 1],
             p + [1, -1],
             p + [-1, 1],
             p + [0, 1],
             p + [0, 1],
             p + [1, 2],
             p + [1, 1],
             p + [0, 2],
             p + [2, 0]]
        )
    )
    expected_distances = [1, 1, 1, 1, 0, np.sqrt(2) / 2, np.sqrt(2), np.sqrt(2), 1, 1]
    np.testing.assert_array_almost_equal(distances, expected_distances)


def test_inscribed_radius():
    # square clockwise
    footprint = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    assert abs(inscribed_radius(footprint)-1.) < 1e-6

    # square anticlockwise
    footprint = np.array([[1, -1], [1, 1], [-1, 1], [-1, -1]])
    assert abs(inscribed_radius(footprint)-1.) < 1e-6

    # square rotated
    footprint = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    assert abs(inscribed_radius(footprint)-np.sqrt(2)/2) < 1e-6

    # triangle
    footprint = np.array([[1, -1], [-1, 1], [-1, -1]])
    assert abs(inscribed_radius(footprint)-0.0) < 1e-6

    footprint = np.array([[2, -1], [-1, 1], [-1, -1]])
    assert abs(inscribed_radius(footprint)-0.27735) < 1e-6

    # square with an appendix
    # The appendix has line segments 0.1 meters close if the distance is define by normals only
    footprint = np.array([[1, -1], [1, 1],
                          [0.1, 1], [0.1, 1.1], [-0.1, 1.1], [-0.1, 1.],
                          [-1, 1], [-1, -1]])
    assert abs(inscribed_radius(footprint)-1.) < 1e-6

    realistic_mule = np.array([
        # left side
        [1.13, 0.33],
        [0.85, 0.33],
        [0.85, 0.4675],
        [0.23, 0.4675],
        [0.23, 0.33],
        [-0.23, 0.33],
        # right side
        [-0.23, -0.33],
        [0.23, -0.33],
        [0.23, -0.4675],
        [0.85, -0.4675],
        [0.85, -0.33],
        [1.13, -0.33],
    ])
    assert abs(inscribed_radius(realistic_mule)-0.23) < 1e-6

    beta_radius = inscribed_radius(_tricycle_footprint())
    assert np.abs(beta_radius - 0.3697) < 1e-3


def test_circumscribed_radius():
    beta_radius = circumscribed_radius(_tricycle_footprint())
    assert abs(beta_radius - 1.348) < 1e-3


def get_pixel_footprint_py(angle, robot_footprint, map_resolution, fill=True):
    '''
    Reference python implementation of get_pixel_footprint
    '''
    assert not isinstance(angle, tuple)
    angles = [angle]
    m = np.empty((2, 2, len(angles)))  # stack of 2 x 2 matrices to rotate the footprint across all desired angles
    c, s = np.cos(angles), np.sin(angles)
    m[0, 0, :], m[0, 1, :], m[1, 0, :], m[1, 1, :] = (c, -s, s, c)
    rot_pix_footprints = np.rollaxis(np.dot(robot_footprint / map_resolution, m), -1)  # n_angles x n_footprint_corners x 2
    # From all the possible footprints, get the outer corner
    footprint_corner = np.maximum(np.amax(rot_pix_footprints.reshape(-1, 2), axis=0),
                                  -np.amin(rot_pix_footprints.reshape(-1, 2), axis=0))
    pic_half_size = np.ceil(footprint_corner).astype(np.int32)
    int_footprints = np.round(rot_pix_footprints).astype(np.int32)
    # get unique int footprints to save time; using http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    flat_int_footprints = int_footprints.reshape(len(angles), -1)
    row_view = np.ascontiguousarray(flat_int_footprints).view(np.dtype((np.void, flat_int_footprints.dtype.itemsize * flat_int_footprints.shape[1])))
    _, idx = np.unique(row_view, return_index=True)
    unique_int_footprints = int_footprints[idx]
    kernel = np.zeros(2 * pic_half_size[::-1] + 1, dtype=np.uint8)
    for f in unique_int_footprints:
        if fill:
            cv2.fillPoly(kernel, [f + pic_half_size], (255, 255, 255))
        else:
            cv2.polylines(kernel, [f + pic_half_size], 1, (255, 255, 255))
    return kernel


def test_get_pixel_footprint_consistency():
    footrpints = [_rectangular_footprint(), _tricycle_footprint()]
    for fill in [True, False]:
        for resolution in [0.03, 0.05]:
            for f in footrpints:
                angles = np.random.random_sample(1000) * np.pi * 2 - np.pi
                for angle in angles:
                    pixel_footprint = get_pixel_footprint(angle, f, resolution, fill=fill)
                    reference = get_pixel_footprint_py(angle, f, resolution, fill=fill)
                    np.testing.assert_array_equal(pixel_footprint, reference)


def test_compute_robot_area():
    footprint = np.array([[-0.77, -0.385], [-0.77, 0.385], [0.67, 0.385], [0.67, -0.385]])
    area = compute_robot_area(0.05, footprint)
    assert area == 493.0


if __name__ == '__main__':
    test_blit()
    test_ensure_float_numpy()
    test_orient_path()
    test_path_velocity()
    test_pose_distances()
    test_distance_to_segments()
    test_distances_to_segment()
    test_distances_to_multiple_segments()
    test_inscribed_radius()
    test_circumscribed_radius()
    test_get_pixel_footprint_consistency()
