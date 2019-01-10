from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from abc import abstractmethod, ABCMeta

from bc_gym_planning_env.robot_models.robot_names import RobotDriveTypes, RobotNames


def get_dimensions_by_name(footprint_name):
    """
    Get class corresponding to footprint_name

    :param footprint_name: footprint name string (see below for valid inputs)
    :return: Dimensions class associated with footprint string, if valid
    """
    name_to_dimensions = {RobotNames.INDUSTRIAL_TRICYCLE_V1: IndustrialTricycleV1Dimensions,
                          RobotNames.INDUSTRIAL_DIFFDRIVE_V1: IndustrialDiffdriveV1Dimensions}

    valid_footprint_types = list(name_to_dimensions.keys())

    if footprint_name not in valid_footprint_types:
        raise AssertionError("Unknown footprint {}. Should be one of {}".format(footprint_name, valid_footprint_types))

    return name_to_dimensions[footprint_name]


def get_drive_type_by_name(platform_name):
    return {RobotNames.INDUSTRIAL_TRICYCLE_V1: RobotDriveTypes.TRICYCLE,
            RobotNames.INDUSTRIAL_DIFFDRIVE_V1: RobotDriveTypes.DIFF
            }[platform_name]


class IDimensions(object):
    """
    Dimensions interface with common abstract static methods to all dimensions classes.
    """

    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def footprint():
        """
        :return: An array of (x,y) points representing the robot's footprint for the given footprint modifier.
        """

    @staticmethod
    @abstractmethod
    def footprint_corner_indices():
        """
        :return: An array of length 4 containing the indices of the footprint array which are the footprint's corners.
        """


class IndustrialDiffdriveV1Dimensions(IDimensions):
    @staticmethod
    def distance_between_wheels():
        """
        returns distance between wheels in meters
        """
        return 0.587  # in meters

    @staticmethod
    def wheel_radius():
        return 0.1524  # meters

    @staticmethod
    def footprint():
        # Note: This is NOT the real footprint, just a mock for the simulator in order to develop a strategy
        footprint = np.array([
            [644.5, 0],
            [634.86, 61],
            [571.935, 130.54],
            [553.38, 161],
            [360.36, 186],
            # Right attachement
            [250, 186],
            [250, 186],
            [100, 186],
            [100, 186],
            # End of right attachement
            [0, 196],
            [-119.21, 190.5],
            [-173.4, 146],
            [-193, 0],
            [-173.4, -143],
            [-111.65, -246],
            [-71.57, -246],
            # Left attachement
            [100, -246],
            [100, -246],
            [250, -246],
            [250, -246],
            # End of left attachement
            [413.085, -223],
            [491.5, -204.5],
            [553, -161],
            [634.86, -62]
        ]) / 1000.

        assert(footprint[0, 1] == 0)  # bumper front-center has to be the first one (just so that everything is correct)
        return footprint

    @staticmethod
    def footprint_height():
        return 1.20

    @staticmethod
    def footprint_corner_indices():
        corner_indices = np.array([4, 11, 17, 24])
        return corner_indices


class IndustrialTricycleV1Dimensions(IDimensions):
    @staticmethod
    def front_wheel_from_axis():
        # Front wheel is 964mm in front of the origin (center of rear-axle)
        return 0.964

    @staticmethod
    def side_wheel_from_axis():
        # Side-wheel touching the ground from the origin (without wheel-cap)
        return 0.3673

    @staticmethod
    def max_front_wheel_angle():
        return 0.5*170*np.pi/180.

    @staticmethod
    def max_front_wheel_speed():
        return 60.*np.pi/180.  # deg per second to radians

    @staticmethod
    def max_linear_acceleration():
        return 1./2.5  # m/s per second. It needs few seconds to achieve 1 m/s speed

    @staticmethod
    def max_angular_acceleration():
        return 1./2.  # rad/s per second. It needs 2 seconds to achieve 1 rad/s rotation speed

    @staticmethod
    def front_column_model_p_gain():
        return 0.16  # P-gain value based on the fitting the RW data to this model

    @staticmethod
    def footprint():
        footprint = np.array([
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
        assert(footprint[0, 1] == 0)  # bumper front-center has to be the first one (just so that everything is correct)
        footprint[:, 1] *= 0.95
        return footprint

    @staticmethod
    def footprint_corner_indices():
        corner_indices = np.array([2, 5, 10, 14])
        return corner_indices
