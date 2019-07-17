"""scan_tools.py
Tools used to manipulate scan data
"""
from __future__ import print_function, absolute_import, division

import numpy as np

from bc_gym_planning_env.utilities.coordinate_transformations import project_points
from bc_gym_planning_env.utilities.numpy_utils import fast_hstack


def scan_to_cartesian(scan_ranges, angles, range_min=0., range_max=25.0):
    """
    Convert range scan to Nx2 array of cartesian coordinates

    :param scan_ranges: 1d array of scan ranges
    :param angles: angle of lidar rays
    :param range_min: filter all ranges below this [meters]
    :param range_max: filter all ranges above this [meters]
    :return: Nx2 array of cartesian scan data
    """
    # Compute scan points in scan coordinate frame
    ranges = np.array(scan_ranges)
    assert ranges.ndim <= 2
    assert angles.ndim == 1
    assert ranges.shape[-1] == angles.shape[0]

    # it is fine to have nans here. Therefore we ignore the invalid values warnings
    with np.errstate(invalid='ignore'):
        bad_ranges = (ranges < range_min) | (ranges > range_max) | np.isinf(ranges)
    masked_ranges = np.copy(scan_ranges)
    masked_ranges[bad_ranges] = np.nan

    unit_vectors = fast_hstack((np.cos(angles.reshape(-1, 1)), np.sin(angles.reshape(-1, 1))))
    xy = unit_vectors * masked_ranges[..., None]

    return xy


def range_scan_to_points(range_scan, origin_pose, range_min=-1., range_max=np.inf):
    """
    Project range scans into world coordinate system assuming that the sensor is at origin_pose.
    Additionally, removes nan ranges (TODO: this seem reduntant since scan_to_cartesian also does that)
    :param range_scan array(2, N)[float64]: array of [[angles, ranges]]
    :param origin_pose array(3)[float64]: (x, y, angle) pose of the scan origin
    :param range_min: filter all ranges below this [meters]
    :param range_max: filter all ranges above this [meters]
    :return array(M, 2)[float64]: M (x, y) points from the scan
    """
    endpoints = scan_to_cartesian(range_scan[1, :], range_scan[0, :], range_min=range_min, range_max=range_max)
    bad_measurements = np.isnan(endpoints[:, 0])
    filtered_endpoints = endpoints[~bad_measurements]
    return project_points(origin_pose, filtered_endpoints)
