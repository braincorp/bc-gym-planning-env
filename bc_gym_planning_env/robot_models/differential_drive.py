from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

from bc_gym_planning_env.utilities.map_drawing_utils import get_pixel_footprint_for_drawing, get_physical_angle_from_drawing
from bc_gym_planning_env.utilities.path_tools import draw_arrow, blit, normalize_angle, path_velocity
from bc_gym_planning_env.robot_models.interface import IRobot
from bc_gym_planning_env.robot_models.robot_names import RobotDriveTypes
from bc_gym_planning_env.utilities.robot_dimensions import get_dimensions_by_name


def kinematic_body_pose_motion_step(pose, linear_velocity, angular_velocity, dt):
    """
    Compute a new pose of the robot based on the previous pose and velocity.

    :param pose: A (n_poses, 3) array of poses.
        The second dimension corresponds to (x_position, y_position, angle)
    :param linear_velocity: An (n_poses, ) Array indicating forward velocity
    :param angular_velocity: An (n_poses, ) Array indicating angular velocity
    :param dt: A float time-step (e.g. 0.1)
    :return: A (n_poses, 3) array of poses after 1 step.
    """
    pose_result = np.array(pose, dtype=float)
    angle = pose[..., 2]
    half_wdt = 0.5*angular_velocity*dt
    v_factor = linear_velocity * dt * np.sinc(half_wdt / np.pi)
    pose_result[..., 0] += v_factor * np.cos(angle + half_wdt)
    pose_result[..., 1] += v_factor * np.sin(angle + half_wdt)
    pose_result[..., 2] = normalize_angle(angle + angular_velocity*dt)

    return pose_result


def _gaussian_noise(variance):
    if variance > 0:
        # normal takes std dev instead of variance
        return np.random.normal(0, np.sqrt(variance))
    else:
        return 0.


def kinematic_body_pose_motion_step_with_noise(pose, linear_velocity, angular_velocity, dt, noise_parameters):
    linear_velocity = linear_velocity + _gaussian_noise(
        noise_parameters['alpha1']*linear_velocity**2 + noise_parameters['alpha2']*angular_velocity**2)
    angular_velocity = angular_velocity + _gaussian_noise(
        noise_parameters['alpha3']*linear_velocity**2 + noise_parameters['alpha4']*angular_velocity**2)
    final_rotation_noise = _gaussian_noise(
        noise_parameters['alpha5']*linear_velocity**2 + noise_parameters['alpha6']*angular_velocity**2)

    new_pose = kinematic_body_pose_motion_step(pose, linear_velocity, angular_velocity, dt)
    new_pose[:, 2] = normalize_angle(new_pose[:, 2] + final_rotation_noise*dt)
    return new_pose


class DiffDriveRobot(IRobot):
    """
    A robot running off a "differential drive".
    i.e. Steering is done by changing the motor signals to the wheels.
    """

    def __init__(self, robots_type_name, footprint_scale=1.0):
        """
        :param footprint_scale: a factor to make the footprint smaller or bigger than the actual
        :param robots_type_name: A string with the robot's type name, which should be the RobotNames enum
        """
        self._x = 0.
        self._y = 0.
        self._angle = 0.
        self._noise_parameters = None
        self._measured_v = 0.
        self._measured_w = 0.
        self._last_pose = (self._x, self._y, self._angle)
        self._footprint_scale = footprint_scale
        self._dimensions = get_dimensions_by_name(robots_type_name)

    def get_drive_type(self):
        return RobotDriveTypes.DIFF

    def get_footprint(self):
        return self._dimensions.footprint()*self._footprint_scale

    def get_dimensions(self):
        return self._dimensions

    def get_footprint_scale(self):
        return self._footprint_scale

    def get_robot_state(self):
        """
        Return current state of the robot
        :return: [measured front wheel angle (None for diff drive), measured linear velocity, measured angular velocity]
        """
        return [None, self._measured_v, self._measured_w]

    def get_pose(self):
        return np.array([self._x, self._y, self._angle])

    def get_default_controls(self):
        return 0., 0

    def set_pose(self, x, y, angle):
        self._x = x
        self._y = y
        self._angle = angle
        self._last_pose = (self._x, self._y, self._angle)
        self._measured_v = 0.
        self._measured_w = 0.

    def step(self, dt, control_signals):
        linear_velocity, angular_velocity = control_signals
        self._x, self._y, self._angle = kinematic_body_pose_motion_step(
            np.array([self._x, self._y, self._angle]),
            linear_velocity, angular_velocity, dt)
        v, w = path_velocity([(0,) + self._last_pose, (dt,) + (self._x, self._y, self._angle)])

        self._measured_v = v[0]
        self._measured_w = w[0]

        self._last_pose = (self._x, self._y, self._angle)

    def draw(self, image, px, py, angle, color, map_resolution):
        """
        Draw robot on the image
        :param image: cv image to draw on
        :param px, py: pixel coordinates of the robot
        :param angle: angle of the robot
        :param color: color to draw with
        :param map_resolution: resolution of the image
        """

        # get_pixel_footprint_for_drawing takes angle in physical coordinates
        kernel = get_pixel_footprint_for_drawing(get_physical_angle_from_drawing(angle),
                                                 self.get_footprint(),
                                                 map_resolution)
        blit(kernel, image, px, py, color)
        arrow_length = 15
        draw_arrow(image, (px, py),
                   (px+int(arrow_length*np.cos(angle)), py+int(arrow_length*np.sin(angle))),
                   color=(0, 0, 255),
                   thickness=2)
