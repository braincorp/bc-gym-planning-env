""" Interface for robot model """
from __future__ import print_function
from __future__ import absolute_import
from abc import abstractmethod, ABCMeta


class IRobot(object):
    """ Interface for the class representing a robot. """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_drive_type(self):
        """
        A string identifying the drive type ('tricycle' or 'diff').  See RobotDriveTypes
        :return str: A string identifying the drive type ('tricycle' or 'diff').  See RobotDriveTypes
        """

    @abstractmethod
    def get_initial_state(self):
        """ Get the default state of the robot. """
        pass

    @abstractmethod
    def get_footprint(self):
        """
        Get footprint: a (n_points x 2) array indicating the perimeter of the robot in real world coordinates.
        :return: A (n_points x 2) array indicating the perimeter of the robot in real world coordinates
        """

    @abstractmethod
    def get_default_controls(self):
        """
        Get the default controls.
        :return: The default controls when there is no solution for the specified local path (too close to the wall)
        """

    @abstractmethod
    def get_footprint_scale(self):
        """
        Note: This method is intended to be used under testing, since real robot's footprints can't be scaled
        :return: The testing footprint scale that was set during the robot's object construction time
        """

    @abstractmethod
    def draw(self, image, px, py, angle, color, map_resolution, alpha=1.0, draw_steering_details=True):
        """
        Draw robot on the image
        :param image: cv image to draw on
        :param px: pixel coordinates of the robot
        :param py: pixel coordinates of the robot
        :param angle: angle of the robot in drawing coordinates
        :param color: color to draw with
        :param map_resolution: resolution of the image
        :param alpha float: transparency of the robot to draw
        :param draw_steering_details bool: Should draw state of steering on the image
        """

    @abstractmethod
    def set_pose(self, x, y, angle):
        """
        set the robot pose.

        :param x float: robot's pose
        :param y float: a robot pose
        :param angle float: a pose of a robot
        """

    @abstractmethod
    def get_robot_state(self):
        """
        Get the internal state of the robot (e.g. wheel angle)
        :return: Get the internal state of the robot (e.g. wheel angle)
        """

    @abstractmethod
    def get_robot_type_name(self):
        """
        Get robot type name
        :return: A string from RobotNames with the robot's type name.
        """
