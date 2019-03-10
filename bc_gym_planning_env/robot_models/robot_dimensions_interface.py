""" Interface for dimensions of different robots"""
from __future__ import print_function
from __future__ import absolute_import
from abc import abstractmethod, ABCMeta


class IDimensions(object):
    """
    Dimensions interface with common abstract static methods to all dimensions classes.
    """
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def footprint(dynamic_footprint_modifier=0.):
        """
        Get footprint - an array of (x,y) points representing the robot's footprint for the given footprint modifier.
        :param dynamic_footprint_modifier: Modifies the footprint in order to obtain a dynamic footprint.
            When the robot doesn't have a dynamic footprint, this can be simply skipped to use a static footprint.
        :return np.ndarray(N, 2): An array of (x,y) points representing the robot's footprint for the given footprint modifier.
        """

    @staticmethod
    @abstractmethod
    def drive_type():
        """
        Return drive type of the robot (e.g. tricycle or diffdrive)
        :return RobotDriveTypes: String from RobotDriveTypes
        """

    @staticmethod
    @abstractmethod
    def get_name():
        """
        A string representing the name dimensions belong to.
        TODO: remove because it creates circular dependency: dimensions can be obtained by name
        TODO: but the name has to be consistent with get_name()
        :return: A string representing the type of robot that the dimensions belong to.
        """


class IFootprintCornerIndices(object):
    """
    Extra characteristics of the footprint robot needed for some planners (e.g. TDWA recovery).
    """

    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def footprint_corner_indices():
        """
        Array of length 4 containing the indices of the footprint array which are the footprint's corners.
        These indices are used to determine side of robot (front, left, right, back) and then might be
        used for planning ("obstacle is on the left, I should move right") or for providing feedback to the user
        There are shapes that are not obvious where the corners should be (e.g. circle or concave shapes)
        so its left up to human to decide for each robot.
        TODO: there must be an algorithm to figure it out though
        :return: An array of length 4 containing the indices of the footprint array which are the footprint's corners.
        """


class IFootprintHeight(object):
    """
    Extra characteristics of the robot height for perception algorithms
    """
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def footprint_height():
        """
        We need footprint height for some perception algorithms to determine if we can safely go below certain
        obstacles
        :return: A float representing the robot's height in meters.
        """


class IDiffdriveDimensions(object):
    """
    Characteristics of a diffdrive robot
    """
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def distance_between_wheels():
        """
        Distance between wheels in meters
        :return float: distance between wheels in meters
        """


class ITricycleDimensions(object):
    """
    Characteristics of a tricycle robot
    """
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def front_wheel_from_axis():
        """
        Distance between center of the rear axis to the front tricycle wheel
        :return float: Distance between center of the rear axis to the front tricycle wheel
        """

    @staticmethod
    @abstractmethod
    def side_wheel_from_axis():
        """
        Distance between center of the rear axis and the back wheel
        :return float: Distance between center of the rear axis and the back wheel, m
        """

    @staticmethod
    @abstractmethod
    def max_front_wheel_angle():
        """
        Max front wheel angle
        :return float: Max front wheel angle
        """

    @staticmethod
    @abstractmethod
    def max_front_wheel_speed():
        """
        Maximum speed of the front wheel rotation in rad/s
        :return float: maximum speed of the front wheel rotation in rad/s
        """

    @staticmethod
    @abstractmethod
    def max_linear_acceleration():
        """
        Max achievable linear acceleration of the robot
        :return float: Max achievable linear acceleration of the robot m/s/s
        """

    @staticmethod
    @abstractmethod
    def max_angular_acceleration():
        """
        Max achievable angular acceleration of the robot
        :return float: Max achievable angular acceleration of the robot rad/s/s
        """

    @staticmethod
    @abstractmethod
    def front_column_model_p_gain():
        """
        P-gain of front wheel P-controller model (e.g. based on the fitting the RW data to simple P model)
        :return float: P-gain of front wheel P-controller model
        """


def assert_footprint_corner_indices_length(corner_indices_array):
    """Common assert for footprint corner indices (there supposed to be 4 corners)
    :param corner_indices_array array(4)[int]: array of indices of corners
    """
    assert len(corner_indices_array) == 4, "You must specify 4 corner indices on the robot's dimensions object."
