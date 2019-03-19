""" Saved costants - dimensions of robots. """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

from bc_gym_planning_env.robot_models.robot_dimensions_interface import IDimensions, ITricycleDimensions, \
    IDiffdriveDimensions, IFootprintCornerIndices, assert_footprint_corner_indices_length
from bc_gym_planning_env.robot_models.standard_robot_names_examples import StandardRobotExamples
from bc_gym_planning_env.robot_models.robot_drive_types import RobotDriveTypes


def get_dimensions_example(footprint_name):
    """
    Get class corresponding to footprint_name

    :param footprint_name: footprint name string (see below for valid inputs)
    :return IDimensions: Dimensions class associated with footprint string, if valid
    """
    name_to_dimensions = {StandardRobotExamples.INDUSTRIAL_TRICYCLE_V1: IndustrialTricycleV1Dimensions,
                          StandardRobotExamples.INDUSTRIAL_DIFFDRIVE_V1: IndustrialDiffdriveV1Dimensions}

    valid_footprint_types = list(name_to_dimensions.keys())

    if footprint_name not in valid_footprint_types:
        raise AssertionError("Unknown footprint {}. Should be one of {}".format(footprint_name, valid_footprint_types))

    return name_to_dimensions[footprint_name]


class IndustrialDiffdriveV1Dimensions(IDimensions, IDiffdriveDimensions, IFootprintCornerIndices):
    """ Dimensions of an industrial differential drive robot. """
    @staticmethod
    def distance_between_wheels():
        """
        returns distance between wheels in meters
        :return: distance between wheels in meters
        """
        return 0.587  # in meters

    @staticmethod
    def wheel_radius():
        """
        Get radius of the wheel in meters.
        :return: radius of the wheel
        """
        return 0.1524  # meters

    @staticmethod
    def footprint(dynamic_footprint_modifier=0.):
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
    def drive_type():
        """
        Return drive type of the robot (e.g. tricycle or diffdrive)
        :return RobotDriveTypes: String from RobotDriveTypes
        """
        return RobotDriveTypes.DIFF

    @staticmethod
    def get_name():
        return StandardRobotExamples.INDUSTRIAL_DIFFDRIVE_V1

    @staticmethod
    def footprint_corner_indices():
        corner_indices = np.array([4, 11, 17, 24])
        assert_footprint_corner_indices_length(corner_indices)
        return corner_indices


class IndustrialTricycleV1Dimensions(IDimensions, ITricycleDimensions, IFootprintCornerIndices):
    """ Dimensions of industrial tricycle """
    @staticmethod
    def front_wheel_from_axis():
        """
        Front wheel is 964mm in front of the origin (center of rear-axle)
        :return float: front wheel to axis distance
        """
        return 0.964

    @staticmethod
    def side_wheel_from_axis():
        """
        Side-wheel touching the ground from the origin (without wheel-cap)
        :return float: Side-wheel touching the ground from the origin (without wheel-cap)
        """
        return 0.3673

    @staticmethod
    def max_front_wheel_angle():
        """
        Maximum front wheel angle
        :return float: Maximum front wheel angle
        """
        return 0.5*170*np.pi/180.

    @staticmethod
    def max_front_wheel_speed():
        """
        deg per second to radians
        :return float: Maximum front wheel speed
        """
        return 60.*np.pi/180.

    @staticmethod
    def max_linear_acceleration():
        """
         Maximum linear acceleration in m/s per second. It needs few seconds to achieve 1 m/s speed
        :return: Maximum linear acceleration
        """
        return 1./2.5

    @staticmethod
    def max_angular_acceleration():
        """
        rad/s per second. It needs 2 seconds to achieve 1 rad/s rotation speed
        :return float: Maximum angular acceleration
        """
        return 1./2.

    @staticmethod
    def front_column_model_p_gain():
        """ P-gain value based on the fitting the RW data to this model.
         :return float: P-gain value based on the fitting the RW data to this model.
         """
        return 0.16

    @staticmethod
    def footprint(dynamic_footprint_modifier=0.):
        """ Get footprint of this robot.
        returns an array of (x,y) points representing the robot's footprint for the given footprint modifier.
        :param dynamic_footprint_modifier: Modifies the footprint in order to obtain a dynamic footprint.
            When the robot doesn't have a dynamic footprint, this can be simply skipped to use a static footprint.
        :return np.ndarray(N, 2): footprint in our internal representation
        """
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
        return footprint

    @staticmethod
    def drive_type():
        return RobotDriveTypes.TRICYCLE

    @staticmethod
    def get_name():
        return StandardRobotExamples.INDUSTRIAL_TRICYCLE_V1

    @staticmethod
    def footprint_corner_indices():
        corner_indices = np.array([2, 5, 10, 14])
        assert_footprint_corner_indices_length(corner_indices)
        return corner_indices
