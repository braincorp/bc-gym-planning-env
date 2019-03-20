from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import numpy as np
from bc_gym_planning_env.robot_models.tricycle_model import diff_drive_control_to_tricycle


def test_diff_drive_control_to_tricycle():
    # straight line, wheel is already straight
    np.testing.assert_almost_equal(
        diff_drive_control_to_tricycle(1.0, 0., 0., np.pi/2, 1.),
        [1., 0.]
    )

    # straight line, wheel is 90 degrees
    np.testing.assert_almost_equal(
        diff_drive_control_to_tricycle(1.0, 0., np.pi/2, np.pi/2, 1.),
        [0., 0.]  # turn the wheel first
    )

    # straight line, wheel is 45 degrees
    np.testing.assert_almost_equal(
        diff_drive_control_to_tricycle(1.0, 0., -np.pi/4, np.pi/2, 1.),
        [0.7071068, 0.]  # turn the wheel first, but start moving
    )

    # rotate anticlockwise
    np.testing.assert_almost_equal(
        diff_drive_control_to_tricycle(0.0, 1., -np.pi/4, np.pi/2, 1.),
        [0., np.pi/2]  # turn the wheel first
    )

    # rotate anticlockwise
    np.testing.assert_almost_equal(
        diff_drive_control_to_tricycle(0.0, 1., np.pi/2-0.1, np.pi/2, 1.),
        [0.9950042, np.pi/2]  # turn the wheel and start moving
    )

    # rotate anticlockwise
    np.testing.assert_almost_equal(
        diff_drive_control_to_tricycle(1.0, 1., np.pi/2-0.1, np.pi/2, 1.),
        [1.0948376, np.pi/4]  # turn the wheel and start moving
    )


if __name__ == '__main__':
    test_diff_drive_control_to_tricycle()
