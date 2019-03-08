""" Logic for differential drive motion. """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import attr

from bc_gym_planning_env.robot_models.robot_drive_types import RobotDriveTypes
from bc_gym_planning_env.utilities.map_drawing_utils import get_pixel_footprint_for_drawing, get_physical_angle_from_drawing
from bc_gym_planning_env.utilities.path_tools import draw_arrow, blit, normalize_angle, path_velocity
from bc_gym_planning_env.robot_models.interface import IRobot


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
    pose_result[..., 2] = normalize_angle(angle + angular_velocity * dt)

    return pose_result


def _gaussian_noise(variance):
    """Generate some gaussian noise of given variance
    :param variance float: variance of the wanted noise
    :return float: the actual 1-d noise float
    """
    if variance > 0:
        # normal takes std dev instead of variance
        return np.random.normal(0, np.sqrt(variance))
    else:
        return 0.


def kinematic_body_pose_motion_step_with_noise(pose, linear_velocity, angular_velocity, dt, noise_parameters):
    """ Execute noisy motion step for an array of poses.
    :param pose: A (n_poses, 3) array of poses.
        The second dimension corresponds to (x_position, y_position, angle)
    :param linear_velocity float: the linear velocity
    :param angular_velocity flaot: the angular velocity
    :param dt float: time interval passing between steps
    :param noise_parameters Dict: dictionary of noise parameters
    :return float np.ndarray(n_poses, 3): array of output posese
    """
    linear_velocity = linear_velocity + _gaussian_noise(
        noise_parameters['alpha1']*linear_velocity**2 + noise_parameters['alpha2']*angular_velocity**2)
    angular_velocity = angular_velocity + _gaussian_noise(
        noise_parameters['alpha3']*linear_velocity**2 + noise_parameters['alpha4']*angular_velocity**2)
    final_rotation_noise = _gaussian_noise(
        noise_parameters['alpha5']*linear_velocity**2 + noise_parameters['alpha6']*angular_velocity**2)

    new_pose = kinematic_body_pose_motion_step(pose, linear_velocity, angular_velocity, dt)
    new_pose[:, 2] = normalize_angle(new_pose[:, 2] + final_rotation_noise * dt)
    return new_pose


@attr.s
class DiffdriveRobotState(object):
    """ State of the diffdrive robot. """
    x = attr.ib(default=0.0, type=float)
    y = attr.ib(default=0.0, type=float)
    angle = attr.ib(default=0.0, type=float)

    v = attr.ib(default=0.0, type=float)
    w = attr.ib(default=0.0, type=float)

    def copy(self):
        """ Return the copy of the state
        :return DiffdriveRobotState: copy of the state
        """
        return attr.evolve(self)

    def get_pose(self):
        """ Get pose part of the state
        :return array(3)[float64]: The pose x, y, angle
        """
        return np.array([self.x, self.y, self.angle])

    def set_pose(self, pose):
        """ Set the pose part of the state
        :param pose np.ndarray: pose of the robot
        """
        self.x, self.y, self.angle = pose

    def to_numpy_array(self):
        """ Render the state to the numpy array.
        :return np.ndarray: the numpy array reflecting the state
        """
        return np.array([self.x, self.y, self.angle, self.v, self.w], dtype=np.float64)


class DiffDriveRobot(IRobot):
    """
    A robot running off a "differential drive".
    i.e. Steering is done by changing the motor signals to the wheels.
    """

    def __init__(self, dimensions, footprint_scale=1.0):
        """
        :param dimensions IDimensions: robot dimensions object
        :param footprint_scale: a factor to make the footprint smaller or bigger than the actual
        """
        self._noise_parameters = None
        self._state = DiffdriveRobotState()

        self._footprint_scale = footprint_scale
        self._dimensions = dimensions

    def get_initial_state(self):
        return DiffdriveRobotState()

    def set_state(self, state):
        """ Set the state of the robot
        :param state DiffdriveRobotState: the state of robot to state
        """
        self._state = state.copy()

    def get_state(self):
        """
        Get state of the robot
        :return DiffdriveRobotState: the state of the robot
        """
        return self._state.copy()

    def get_drive_type(self):
        return RobotDriveTypes.DIFF

    def get_footprint(self):
        return self._dimensions.footprint()*self._footprint_scale

    def get_dimensions(self):
        """ Get dimensions of the robot
        :return IDimensions: Dimensions of the robot """
        return self._dimensions

    def get_footprint_scale(self):
        return self._footprint_scale

    def get_robot_state(self):
        """
        Return current state of the robot
        :return: [measured front wheel angle (None for diff drive), measured linear velocity, measured angular velocity]
        """
        return [None, self._state.v, self._state.w]

    def get_pose(self):
        """
        Get the pose of the robot.
        :return np.ndarray(3): pose of the robot (x, y, angle)
        """
        return self._state.get_pose()

    def get_default_controls(self):
        return 0., 0

    def set_pose(self, x, y, angle):
        """
        Set the current pose of the robot
        :param x float: x coordinate of the robot pose
        :param y float: y coordinate of the robot pose
        :param angle float: angle from 0 to 2pi of the robot pose
        """
        self._state.x = x
        self._state.y = y
        self._state.angle = angle
        self._state.v = 0.0
        self._state.w = 0.0

    def step(self, dt, action):
        """
        Make a step, applying the control signals
        :param dt float: Time interval that passes between the time steps
        :param action Action: motion primitive to execute in this environment state
            for diffdrive command in the action is (v, w)
        """
        last_pose = self._state.get_pose()
        linear_velocity, angular_velocity = action.command
        if self._noise_parameters is None:
            new_pose = kinematic_body_pose_motion_step(self._state.get_pose(),
                linear_velocity, angular_velocity, dt)
        else:
            new_pose = kinematic_body_pose_motion_step_with_noise(
                self._state.get_pose(), linear_velocity, angular_velocity, dt, self._noise_parameters)

        v, w = path_velocity(
            np.vstack([
                np.hstack([
                    np.array(0), last_pose
                ]),
                np.hstack([
                    np.array(dt), new_pose
                ])
            ])
        )

        self._state.set_pose(new_pose)
        self._state.v = v[0]
        self._state.w = w[0]

    def draw(self, image, px, py, angle, color, map_resolution, alpha=1.0, draw_steering_details=True):
        """
        Draw robot on the image
        :param image: cv image to draw on
        :param px: pixel coordinates of the robot
        :param py: pixel coordinates of the robot
        :param angle: angle of the robot
        :param color: color to draw with
        :param map_resolution: resolution of the image
        :param alpha float: transparency of the robot image
        :param draw_steering_details bool: Should draw state of steering on the image
        """

        # get_pixel_footprint_for_drawing takes angle in physical coordinates
        kernel = get_pixel_footprint_for_drawing(get_physical_angle_from_drawing(angle),
                                                 self.get_footprint(),
                                                 map_resolution)
        blit(kernel, image, px, py, color, alpha=alpha)
        arrow_length = 15
        draw_arrow(image, (px, py),
                   (px+int(arrow_length*np.cos(angle)), py+int(arrow_length*np.sin(angle))),
                   color=(0, 0, 255),
                   thickness=2)

    def set_noise_parameters(self, noise_parameters):
        """
        Set parameters of the odometry noise model
        :param noise_parameters Dict[string, float]: dict of noise parameters for kinematic_body_pose_motion_step_with_noise
        """
        self._noise_parameters = noise_parameters

    def get_robot_type_name(self):
        return self._dimensions.get_name()
