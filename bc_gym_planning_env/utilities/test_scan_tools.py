from __future__ import print_function, absolute_import, division

import numpy as np

from bc_gym_planning_env.utilities.scan_tools import scan_to_cartesian, range_scan_to_points


def test_scan_to_cartesian():
    scan_ranges = np.array([0., 1, 2, 3])
    angles = np.array([-np.pi / 2.-0.2, -np.pi / 2., 0, np.pi / 2.])
    points = scan_to_cartesian(scan_ranges, angles, range_min=0., range_max=25.0)
    np.testing.assert_array_almost_equal(
        points,
        [[0., 0.],
         [0., -1.],
         [2., 0.],
         [0., 3]]
    )

    scan_ranges = np.array([3., 2, 1, 3])
    angles = np.array([-np.pi / 2., -np.pi / 2., 0, np.pi / 2.])
    points = scan_to_cartesian(scan_ranges, angles, range_min=0., range_max=25.0)
    np.testing.assert_array_almost_equal(
        points,
        [[0., -3.],
         [0., -2.],
         [1., 0.],
         [0., 3]]
    )

    scan_ranges = np.array([0., 1, 2, 3])
    angles = np.array([-np.pi / 2. - 0.2, -np.pi / 2., 0, np.pi / 2.])
    points = scan_to_cartesian(scan_ranges, angles, range_min=0.1, range_max=2.5)
    np.testing.assert_array_almost_equal(
        points,
        [[np.nan, np.nan],
         [0., -1.],
         [2., 0.],
         [np.nan, np.nan]]
    )


def test_scan_to_cartesian_batch():
    scan_ranges = np.array([[0., 1, 2, 3],
                            [3., 2, 1, 3]])
    angles = np.array([-np.pi / 2., -np.pi / 2., 0, np.pi / 2.])
    points = scan_to_cartesian(scan_ranges, angles, range_min=0., range_max=25.0)
    np.testing.assert_array_almost_equal(
        points,
        [[[0., 0.],
          [0., -1.],
          [2., 0.],
          [0., 3]],
         [[0., -3.],
          [0., -2.],
          [1., 0.],
          [0., 3]]]
    )

    scan_ranges = np.array([[0., 1, 2, 3],
                            [3., 2, 1, 3]])
    angles = np.array([-np.pi / 2., -np.pi / 2., 0, np.pi / 2.])
    points = scan_to_cartesian(scan_ranges, angles, range_min=0.1, range_max=2.5)
    np.testing.assert_array_almost_equal(
        points,
        [[[np.nan, np.nan],
          [0., -1.],
          [2., 0.],
          [np.nan, np.nan]],
         [[np.nan, np.nan],
          [0., -2.],
          [1., 0.],
          [np.nan, np.nan]]]
    )


def test_range_scan_to_points():
    np.testing.assert_array_almost_equal(
        range_scan_to_points(
            np.array([[0.], [0.]]),
            np.array([0., 0., 0.])),
        np.array([[0., 0.]])
    )

    np.testing.assert_array_almost_equal(
        range_scan_to_points(
            np.array([[0., np.pi / 2.], [0., 1]]),
            np.array([0., 0., 0.])),
        np.array([
            [0., 0.],
            [0., 1.]])
    )

    np.testing.assert_array_almost_equal(
        range_scan_to_points(
            np.array([[0., np.pi / 2.], [1., 1]]),
            np.array([0., 0., np.pi / 2.])),
        np.array([
            [0., 1.],
            [-1., 0.]])
    )

    np.testing.assert_array_almost_equal(
        range_scan_to_points(
            np.array([[0., np.pi / 2.], [1., 1]]),
            np.array([1., -2., np.pi / 2.])),
        np.array([
            [1., -1.],
            [0., -2.]])
    )

    # remove nan measurements
    np.testing.assert_array_almost_equal(
        range_scan_to_points(
            np.array([[0., np.pi / 2.], [np.nan, 1]]),
            np.array([1., -2., np.pi / 2.])),
        np.array([
            [0., -2.]])
    )

    # remove inf measurements
    np.testing.assert_array_almost_equal(
        range_scan_to_points(
            np.array([[0., np.pi / 2.], [1., np.inf]]),
            np.array([1., -2., np.pi / 2.])),
        np.array([
            [1., -1.]])
    )

    # remove all measurements
    result = range_scan_to_points(
        np.array([[0., np.pi / 2.], [np.inf, np.inf]]),
        np.array([1., -2., np.pi / 2.]))
    assert len(result) == 0

    # remove measurements that are not in the specified range
    np.testing.assert_array_almost_equal(
        range_scan_to_points(
            np.array([[0., 1. * np.pi / 3., 2. * np.pi / 3., np.pi], [6, 4, np.inf, np.nan]]),
            np.array([1., -2., np.pi / 2.]),
            range_min=5, range_max=10),
        np.array([[1., 4.]])
    )


if __name__ == '__main__':
    test_scan_to_cartesian()
    test_scan_to_cartesian_batch()
    test_range_scan_to_points()
