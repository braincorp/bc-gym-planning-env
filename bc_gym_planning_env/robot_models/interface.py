from __future__ import print_function
from __future__ import absolute_import
from abc import abstractmethod, ABCMeta


class IRobot(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_drive_type(self):
        """
        :return: A string identifying the drive type ('tricycle' or 'diff').  See RobotDriveTypes
        """

    @abstractmethod
    def get_footprint(self):
        """
        :return: A (n_points x 2) array indicating the perimeter of the robot in real world coordinates
        """

    @abstractmethod
    def get_default_controls(self):
        """
        :return: The default controls when there is no solution for the specified local path (too close to the wall)
        """

    @abstractmethod
    def get_footprint_scale(self):
        """
        Note: This method is intended to be used under testing, since real robot's footprints can't be scaled
        :return: The testing footprint scale that was set during the robot's object construction time
        """

    @abstractmethod
    def draw(self, image, px, py, angle, color, map_resolution):
        """
        Draw robot on the image
        :param image: cv image to draw on
        :param px, py: pixel coordinates of the robot
        :param angle: angle of the robot in drawing coordinates
        :param color: color to draw with
        :param map_resolution: resolution of the image
        """

    @abstractmethod
    def set_pose(self, x, y, angle):
        """
        :param x, y, angle: A robot pose
        """

    @abstractmethod
    def get_robot_state(self):
        """
        :return: Get the internal state of the robot (e.g. wheel angle)
        """
