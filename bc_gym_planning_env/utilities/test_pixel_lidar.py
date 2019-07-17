from __future__ import print_function, absolute_import, division


from builtins import range
import numpy as np
import logging

import cv2

from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.utilities.pixel_lidar import PixelLidar
from bc_gym_planning_env.utilities.scan_tools import range_scan_to_points


def test_pixel_lidar_orientation():
    lidar = PixelLidar(100, angle_range=np.pi, n_angles=360)
    rng = np.random.RandomState(0)
    im = np.zeros((400, 400), dtype=np.uint8)
    for _ in range(5):
        im[:] = 0
        p = rng.randint(100, 301, size=(2,))
        a = np.int(np.round(rng.rand() * 180))
        r1 = rng.randint(25, 35)
        r2 = rng.randint(65, 85)
        cv2.ellipse(im, tuple(p), (r1, r1), 0, a - 45, a + 45, 254, 1, 4)
        cv2.ellipse(im, tuple(p), (r2, r2), 0, a - 215, a - 145, 254, 1, 4)
        real_ranges = np.zeros(360)
        real_ranges[:] = np.NAN
        r1_range = np.arange(a - 45, a + 45 + 1)
        r2_range = np.arange(a - 215, a - 145 + 1)
        real_ranges[np.mod(r1_range, 360)] = r1
        real_ranges[np.mod(r2_range, 360)] = r2
        for orientation in np.linspace(0, np.pi * 2, 20):
            range_scan, clearing_scan = lidar.get_scan_ranges(im, p, orientation)
            orientation_shift = 180 - np.int(np.round(orientation * 180 / np.pi))
            expected_ranges = np.roll(real_ranges, orientation_shift)[90:271]
            expected_angles = np.arange(-90, 91)
            assert np.all(expected_angles == np.round(range_scan[0, :] * 180 / np.pi).astype(np.int))
            assert np.count_nonzero(np.isnan(range_scan[1, :]) != np.isnan(expected_ranges)) <= 3
            both_ranges = ~np.isnan(range_scan[1, :]) & ~np.isnan(expected_ranges)
            assert np.all(np.abs(range_scan[1, :][both_ranges] - expected_ranges[both_ranges]) <= 1.5)

            assert np.all(np.isinf(clearing_scan[1, :]) == ~np.isnan(range_scan[1, :]))
            assert np.all(clearing_scan[1, ~np.isinf(clearing_scan[1, :])] == 100)


def test_pixel_lidar():
    im = np.zeros((400, 400), dtype=np.uint8)
    rng = np.random.RandomState(0)
    lidar_0 = PixelLidar(100)  # this lidar will find the point at the origin
    lidar_1 = PixelLidar(100, None, blind_radius=1)  # this lidar won't
    lidar_2 = PixelLidar(30, None, blind_radius=1)  # this lidar has a short range
    lidar_3 = PixelLidar(100, np.pi, blind_radius=1)  # this lidar only sees a 180 degree range
    for noise in [False, True]:
        for _ in range(5):
            im[:] = 0
            p = rng.randint(100, 301, size=(2,))
            r = rng.randint(35, 55)
            if noise:
                noisy_points = rng.randint(0, 400, size=(20000, 2))
                im[noisy_points[:, 0], noisy_points[:, 1]] = 254
                cv2.circle(im, tuple(p), r, (0, 0, 0), -1, 4)
            im[p[1], p[0]] = 254
            cv2.circle(im, tuple(p), r, (254, 254, 254), 1)

            for orientation in np.linspace(0, np.pi * 2, 20):
                points_0, clearing_points_0 = lidar_0.get_scan_points(im, p, orientation)
                range_scan_0, clearing_range_scan_0 = lidar_0.get_scan_ranges(im, p, orientation)
                assert len(clearing_points_0) == 0
                assert points_0.shape == (1, 2)
                assert np.all(points_0[0] == p)
                assert np.all(range_scan_0[1, :] == 0.)
                assert np.all(np.isinf(clearing_range_scan_0[1, :]))

                points_1, clearing_points_1 = lidar_1.get_scan_points(im, p, orientation)
                range_scan_1, clearing_range_scan_1 = lidar_1.get_scan_ranges(im, p, orientation)
                assert len(clearing_points_1) == 0
                expected_im = np.zeros_like(im)
                cv2.circle(expected_im, tuple(p), r, (254, 254, 254), 1)
                actual_im = np.zeros_like(im)
                actual_im[points_1[:, 1], points_1[:, 0]] = 254
                assert np.all(expected_im == actual_im)
                assert np.all(np.abs(range_scan_1[1, :] - r) < 1.)
                assert np.all(np.isinf(clearing_range_scan_1[1, :]))

                points_2, clearing_points_2 = lidar_2.get_scan_points(im, p, orientation)
                range_scan_2, clearing_range_scan_2 = lidar_2.get_scan_ranges(im, p, orientation)
                assert points_2.size == 0
                assert np.all(np.abs(np.hypot(clearing_points_2[:, 0] - p[0], clearing_points_2[:, 1] - p[1]) - 30) < 1.0)
                assert np.all(np.isnan(range_scan_2[1, :]))
                assert np.all(clearing_range_scan_2[1, :] == 30)

                points_3, clearing_points_3 = lidar_3.get_scan_points(im, p, orientation)  # we expect half a circle
                range_scan_3, clearing_range_scan_3 = lidar_3.get_scan_ranges(im, p, orientation)
                assert len(clearing_points_3) == 0
                # draw half-circle by removing the undetected half
                expected_im = np.zeros_like(im)
                cv2.circle(expected_im, tuple(p), r, (254, 254, 254), 1)
                y, x = np.indices(im.shape)
                expected_im[np.cos(orientation) * (x - p[0]) + np.sin(orientation) * (y - p[1]) < 0] = 0
                actual_im = np.zeros_like(im)
                actual_im[points_3[:, 1], points_3[:, 0]] = 254
                assert np.count_nonzero(expected_im != actual_im) < 3  # at most 2 extra points
                assert np.all(np.abs(range_scan_3[1, :] - r) < 1.)
                assert np.all(np.isinf(clearing_range_scan_3[1, :]))


def test_pixel_lidar_custom_angles(debug=False):
    costmap = np.zeros((400, 400), dtype=np.uint8)
    lidar = PixelLidar(pixel_radius=200,
                       angle_range=np.pi*2,
                       blind_radius=0,
                       n_angles=360)

    circle_p = [150, 150]
    cv2.circle(costmap, tuple(circle_p), 40, CostMap2D.LETHAL_OBSTACLE, 1)
    lidar_pose = [100, 100]

    orientation = np.pi/4

    range_scan, _ = lidar.get_scan_ranges(
        costmap,
        lidar_pose,
        orientation,
        lidar_angles=np.array([0.5, 0.2, 0.1, 0., -0.1, -0.3, -0.55]))

    points = range_scan_to_points(range_scan, np.array(lidar_pose + [orientation])).astype(int)

    np.testing.assert_array_equal(
        points,
        [[111, 140],
         [117, 126],
         [119, 124],
         [121, 121],
         [124, 119],
         [129, 115],
         [145, 110]]
    )

    if debug:
        costmap[costmap>0] = 50
        costmap[lidar_pose[1], lidar_pose[0]] = 150
        costmap[points[:, 1], points[:, 0]] = 255
        cv2.imshow('a', np.flipud(costmap))
        cv2.waitKey(-1)


def test_pixel_lidar_custom_angles_warparound():
    '''
    There was a bug that led to reading uninitialized memory (stochastically) with a particular set of custom lidar angles.
    Here we check that raytraicing with such angles always gives the same result
    '''
    costmap = np.zeros((200, 200), dtype=np.uint8)

    circle_p = [100, 100]
    cv2.circle(costmap, tuple(circle_p), 40, CostMap2D.LETHAL_OBSTACLE, 1)
    lidar_pose = [100, 100]

    expected_range = np.array(
        [39.849716,
         39.408119, 39.408119, 39.408119, 39.408119, 39.408119, 39.408119,
         40., 40., 40., 40., 40., 40., 40.,
         39.408119, 39.408119, 39.408119, 39.408119, 39.408119, 39.408119])

    for _ in range(1000):
        lidar = PixelLidar(pixel_radius=300,
                           angle_range=np.pi * 2,
                           blind_radius=0,
                           n_angles=10)

        range_scan, _=lidar.get_scan_ranges(
            costmap,
            lidar_pose,
            0.,
            lidar_angles=np.arange(-1, 1, 0.1))

        np.testing.assert_array_almost_equal(expected_range, range_scan[1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_pixel_lidar()
    test_pixel_lidar_orientation()
    test_pixel_lidar_custom_angles()
    test_pixel_lidar_custom_angles_warparound()
    logging.info('Passed all tests!')
