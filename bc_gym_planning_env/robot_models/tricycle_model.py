""" Tricycle drive robot dynamics model. """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import attr
import numpy as np

from shining_software.env_utils.robot_models.differential_drive import \
    kinematic_body_pose_motion_step, kinematic_body_pose_motion_step_with_noise
from shining_software.env_utils.robot_models.robot_interface import IRobot
from shining_software.env_utils.robot_models.robot_drive_types import RobotDriveTypes
from shining_software.env_utils.utilities.map_drawing_utils import \
    get_pixel_footprint_for_drawing, get_physical_angle_from_drawing, puttext_centered
from shining_software.env_utils.utilities.path_tools import blit, draw_arrow, path_velocity


def vw_from_front_wheel_velocity(front_wheel_velocity, front_wheel_angle, front_wheel_from_axis):
    """
    Compute linear and angular velocity of the back wheels center based on front wheel velocity
    and angle

    :param front_wheel_velocity: front_wheel_velocity
    :param front_wheel_angle: front_wheel_angle
    :param front_wheel_from_axis: front_wheel_from_axis
    :return Tuple[float, float]: linear velocity, angular velocity
    """
    v = front_wheel_velocity * np.cos(front_wheel_angle)
    w = front_wheel_velocity * np.sin(front_wheel_angle) / front_wheel_from_axis
    return v, w


def tricycle_kinematic_step(pose, current_wheel_angle, dt, control_signals, max_front_wheel_angle,
                            front_wheel_from_axis,
                            max_front_wheel_speed, front_column_p_gain, model_front_column_pid=True):
    '''
    Integrate tricycle kinematics based on control signals (tricycle kinematic forward model)
    :param pose array(..., 3)[float64]: array of initial poses to integrate forward (x, y, angle)
    :param current_wheel_angle array(..., 1)[float64]: array initial wheel angles
    :param dt: integration timestep
    :param control_signals array(..., 2): array of controls (wheel_v, wheel_angle)
    :param max_front_wheel_angle float: maximum front wheel angle position to clip
    :param front_wheel_from_axis float: distance from the center of rear axis to the front wheel in meters
    :param max_front_wheel_speed float: maximum speed of the front wheel rotation in rad/s
    :param front_column_p_gain float: P-gain of front wheel P-controller model
    :param model_front_column_pid bool: whether to model front wheel P-controller or turn the wheel instantaneously
    :return (array(..., 3)[float64], array(..., 1)[float64]): array of new poses and wheel angles after one timestep
    '''
    # rotate the front wheel first
    if model_front_column_pid:
        new_wheel_angle = tricycle_front_wheel_column_step(
            current_wheel_angle, control_signals[:, 1],
            max_front_wheel_angle, max_front_wheel_speed, front_column_p_gain,
            dt
        )
    else:
        new_wheel_angle = np.clip(control_signals[:, 1], -max_front_wheel_angle, max_front_wheel_angle)

    desired_wheel_v = control_signals[:, 0]
    linear_velocity = desired_wheel_v * np.cos(new_wheel_angle)
    angular_velocity = desired_wheel_v * np.sin(new_wheel_angle) / front_wheel_from_axis

    pose_result = kinematic_body_pose_motion_step(pose, linear_velocity, angular_velocity, dt)
    return pose_result, new_wheel_angle


def tricycle_dynamic_step(
        pose, current_wheel_angle, current_v, current_w, dt, control_signals,
        max_front_wheel_angle, front_wheel_from_axis, max_front_wheel_speed,
        max_linear_acceleration, max_angular_acceleration, front_column_p_gain,
        noise_parameters=None, model_front_column_pid=True):
    '''
    Extension of tricycle_kinematic_step that takes into account inertia and pid on the front wheel column.
    Since it takes measurement v, w of the body of the robot that might not take into account non-holonomic constraints
    of the front wheel velocity, it makes the front wheel slip because of inertia.

    This is not really a proper dynamic model of wheel slippage -
     - it is just an application of inertia constraints to a unconstrainted 2d body and
    then changing v, w based on the wheel motor commands.

    :param pose: ... x 3  (x, y, angle)
    :param current_wheel_angle: ... x 1
    :param current_v: ... x 1
    :param current_w: ... x 1
    :param dt: integration timestep
    :param control_signals: ... x 2 (wheel_v, wheel_angle) controls
    :param max_front_wheel_angle: parameter of the model
    :param front_wheel_from_axis:  parameter of the model
    :param max_front_wheel_speed:  parameter of the model
    :param max_linear_acceleration:  parameter of the model
    :param max_angular_acceleration:  parameter of the model
    :param front_column_p_gain: front_column_p_gain
    :param noise_parameters: None or dictionary of 6 alpha parameters of diff drive noise
        (implements the odometry motion model as described in Thrun et al (sec. 5.4))
    :param model_front_column_pid bool: whether to model PID
    :return: ... x 3 pose (x, y, angle), ... x 1 current_v, ... x 1 current_w, , ... x 1 new_wheel_angle) after one timestep
    '''
    # rotate the front wheel first
    if model_front_column_pid:
        new_wheel_angle = tricycle_front_wheel_column_step(
            current_wheel_angle, control_signals[:, 1],
            max_front_wheel_angle, max_front_wheel_speed, front_column_p_gain,
            dt
        )
    else:
        new_wheel_angle = np.clip(control_signals[:, 1], -max_front_wheel_angle, max_front_wheel_angle)

    new_linear_velocity, new_angular_velocity = tricycle_velocity_dynamic_model_step(
        current_v, current_w, new_wheel_angle, control_signals[:, 0],
        front_wheel_from_axis, max_linear_acceleration, max_angular_acceleration, dt
    )

    if noise_parameters is None:
        pose_result = kinematic_body_pose_motion_step(
            pose, new_linear_velocity, new_angular_velocity, dt)
    else:
        pose_result = kinematic_body_pose_motion_step_with_noise(
            pose, new_linear_velocity, new_angular_velocity, dt, noise_parameters)

    return pose_result, new_wheel_angle, new_linear_velocity, new_angular_velocity


def tricycle_front_wheel_column_step(current_front_wheel_angle, desired_front_wheel_angle,
                                     max_front_wheel_angle, max_front_wheel_speed, front_column_p_gain,
                                     dt):
    """
    The model of the front wheel column which includes PID and takes into account the constraints
    on the angular velocity of the wheel and maximum angle

    :param current_front_wheel_angle: current front wheel angle
    :param desired_front_wheel_angle: desired front wheel angle
    :param max_front_wheel_angle: max front wheel angle
    :param max_front_wheel_speed: max front wheel speed
    :param front_column_p_gain: front column p gain
    :param dt: what is the time interval between the time steps
    :return: new front wheel angle
    """
    # rotate the front wheel first emulating a pid controller on the front wheel with a finite rotation speed
    max_front_wheel_delta = max_front_wheel_speed * dt
    clip_first = False
    if clip_first:
        desired_wheel_delta = desired_front_wheel_angle - current_front_wheel_angle
        desired_wheel_delta = np.clip(desired_wheel_delta, -max_front_wheel_delta, max_front_wheel_delta)
        new_front_wheel_angle = current_front_wheel_angle + front_column_p_gain * desired_wheel_delta
    else:
        desired_wheel_delta = front_column_p_gain * (desired_front_wheel_angle - current_front_wheel_angle)
        desired_wheel_delta = np.clip(desired_wheel_delta, -max_front_wheel_delta, max_front_wheel_delta)
        new_front_wheel_angle = current_front_wheel_angle + desired_wheel_delta
    new_front_wheel_angle = np.clip(new_front_wheel_angle, -max_front_wheel_angle, max_front_wheel_angle)
    return new_front_wheel_angle


def tricycle_velocity_dynamic_model_step(
        current_v, current_w, current_wheel_angle, desired_wheel_v,
        front_wheel_from_axis, max_linear_acceleration, max_angular_acceleration, dt):
    """
    Calculate next step linear and angular velocity
    :param current_v: current line velocity
    :param current_w: current angular velocity
    :param current_wheel_angle:  current wheel angle
    :param desired_wheel_v:  wanted wheel angle
    :param front_wheel_from_axis: how far is the front wheel from axis, m
    :param max_linear_acceleration: what is maximum linear acceleration
    :param max_angular_acceleration: what is maximum angular acceleration
    :param dt: what is the time interval between the time steps
    :return Tuple[float, float]: linear velocity, angular velocity
    """

    desired_linear_velocity = desired_wheel_v * np.cos(current_wheel_angle)
    desired_angular_velocity = desired_wheel_v * np.sin(current_wheel_angle) / front_wheel_from_axis

    desired_linear_acceleration = (desired_linear_velocity - current_v) / dt
    desired_angular_acceleration = (desired_angular_velocity - current_w) / dt

    # we can decelerate twice as fast as accelerate
    linear_acceleration = np.clip(desired_linear_acceleration, -2 * max_linear_acceleration, max_linear_acceleration)
    angular_acceleration = np.clip(desired_angular_acceleration, -max_angular_acceleration, max_angular_acceleration)

    new_linear_velocity = (current_v + linear_acceleration * dt)
    new_angular_velocity = (current_w + angular_acceleration * dt)

    return new_linear_velocity, new_angular_velocity


@attr.s
class TricycleRobotState(object):
    """ State of the tricycle robot. """
    x = attr.ib(default=0.0, type=float)
    y = attr.ib(default=0.0, type=float)
    angle = attr.ib(default=0.0, type=float)

    v = attr.ib(default=0.0, type=float)
    w = attr.ib(default=0.0, type=float)
    steering_motor_command = attr.ib(default=0.0, type=float)
    wheel_angle = attr.ib(default=0.0, type=float)

    def copy(self):
        """ Return the copy of the state
        :return TricycleRobotState: copy of the state
        """
        return attr.evolve(self)

    def get_pose(self):
        """ Get pose part of the state
        :return Tuple[float, float, float]: The pose x, y, angle
        """
        return self.x, self.y, self.angle

    def set_pose(self, pose):
        """ Set the pose part of the state
        :param pose np.ndarray: pose of the robot
        """
        self.x, self.y, self.angle = pose

    def to_numpy_array(self):
        """ Render the state to the numpy array.
        :return np.ndarray: the numpy array reflecting the state
        """
        return np.array([self.x, self.y, self.angle, self.v, self.w, self.wheel_angle], dtype=np.float64)

    def old_style(self):
        """
        Get the state of the robot.
        :return List[float]: (wheel_angle, measured_v, measured_w, steering_motor_command)
        """
        return [self.wheel_angle, self.v, self.w, self.steering_motor_command]


class TricycleRobot(IRobot):
    """ Tricycle drive robot. """

    def __init__(self, dimensions, footprint_scale=1.0,
                 wheel_angle=0.,
                 model_front_column_pid=True,
                 dynamic_model=True
                 ):
        '''
        :param dimensions object: dimensions object
        :param footprint_scale: a factor to make the footprint smaller or bigger than the actual
        :param wheel_angle float:  wheel angle
        :param model_front_column_pid bool: turn on PID modeling or assume immediate steering wheel rotation
        :param dynamic_model bool: use dynamic model with mass or a kinematic one
        '''
        self._footprint_scaling = footprint_scale
        self._noise_parameters = None
        self._model_front_column_pid = model_front_column_pid
        self._dynamic_model = dynamic_model
        self._dimensions = dimensions
        self._state = TricycleRobotState(wheel_angle=wheel_angle)

        # TODO: Shoud be in the state (but in some other abstraction)
        self._scrubdeck_status = {
            'vacuum': False,
            'brushes': False
        }

    def get_initial_state(self):
        return TricycleRobotState()

    def set_state(self, state):
        """ Set the state of the robot
        :param state TricycleRobotState: the state of robot to state
        """
        self._state = state.copy()

    def get_state(self):
        """
        Get state of the robot
        :return TricycleRobotState: the state of the robot
        """
        return self._state.copy()

    def get_drive_type(self):
        return RobotDriveTypes.TRICYCLE

    def get_footprint(self):
        footprint = self._dimensions.footprint() * self._footprint_scaling
        return footprint

    def get_footprint_scale(self):
        """
        Get scale of the footprint.
        :return float: Scale of the footprint
        """
        return self._footprint_scaling

    def get_pose(self):
        """
        Get the pose of the robot.
        :return: pose of the robot
        """
        return np.array([
            self._state.x,
            self._state.y,
            self._state.angle
        ])

    def get_front_wheel_angle(self):
        """
        Get angle of the front wheel.
        :return float: Front wheel angle.
        """
        return self._state.wheel_angle

    def get_robot_state(self):
        """
        Get the state of the robot, that is unrelated to its pose.
        TODO: this is a legacy method, refactor users to use just state (old_style)
        :return array(4)[float64]: state of the robot
        """
        return np.array([
            self._state.wheel_angle,
            self._state.v,
            self._state.w,
            self._state.steering_motor_command
        ])

    def set_front_wheel_angle(self, angle):
        """
        Set wheel angle
        :param angle float:  the wheel angle
        """
        self._state.wheel_angle = angle

    def get_dimensions(self):
        """
        Get dimensions.
        :return IDimensions: dimensions of the model
        """
        return self._dimensions

    def get_front_wheel_from_axis_distance(self):
        """
        Get front wheel from axis distance
        :return float: Get front wheel from axis distance
        """
        return self._dimensions.front_wheel_from_axis()

    def get_max_front_wheel_angle(self):
        """
        Get max front wheel angle.
        :return float: Get max front wheel angle.
        """
        return self._dimensions.max_front_wheel_angle()

    def get_max_front_wheel_speed(self):
        """
        Get maximum front wheel speed.
        :return float: Get maximum front wheel speed.
        """
        return self._dimensions.max_front_wheel_speed()

    def get_max_linear_acceleration(self):
        """
        Get maximum linear acceleration.
        :return float: Get maximum linear acceleration.
        """
        return self._dimensions.max_linear_acceleration()

    def get_max_angular_acceleration(self):
        """
        Max angular acceleration.
        :return float: Max angular acceleration
        """
        return self._dimensions.max_angular_acceleration()

    def get_front_column_model_p_gain(self):
        """
        Front column model p gain.
        :return float: Front column model p gain
        """
        return self._dimensions.front_column_model_p_gain()

    def get_default_controls(self):
        return 0., self.get_front_wheel_angle()

    def set_pose(self, x, y, angle):
        self._state.x = x
        self._state.y = y
        self._state.angle = angle
        self._state.v = 0.0
        self._state.w = 0.0

    def step(self, dt, action):
        """
        step when control signals are wheel linear speed and wheel angle

        :param dt float: period between timesteps
        :param action Action: motion primitive to execute in this environment state
            for tricycle command in the action is (v, desired_wheel_angle)
        """
        last_pose = self._state.get_pose()

        if self._dynamic_model:
            new_poses, new_wheel_angles, _, _ = \
                tricycle_dynamic_step(
                    np.array([list(last_pose)]),
                    [self._state.wheel_angle],
                    [self._state.v],
                    [self._state.w],
                    dt,
                    np.array([action.command]),
                    self.get_max_front_wheel_angle(),
                    self.get_front_wheel_from_axis_distance(),
                    self.get_max_front_wheel_speed(),
                    self.get_max_linear_acceleration(),
                    self.get_max_angular_acceleration(),
                    self.get_front_column_model_p_gain(),
                    self._noise_parameters,
                    model_front_column_pid=self._model_front_column_pid
                )
        else:
            new_poses, new_wheel_angles = \
                tricycle_kinematic_step(
                    np.array([list(last_pose)]),
                    [self._state.wheel_angle],
                    dt,
                    np.array([action.command]),
                    self.get_max_front_wheel_angle(),
                    self.get_front_wheel_from_axis_distance(),
                    self.get_max_front_wheel_speed(),
                    self.get_front_column_model_p_gain(),
                    model_front_column_pid=self._model_front_column_pid
                )

        v, w = path_velocity(
            np.vstack([
                np.hstack([
                    np.array(0), last_pose
                ]),
                np.hstack([
                    np.array(dt), new_poses[0]
                ])
            ])
        )
        # This is a very crude estimation of the steering motor command angle based on the current angle of the wheel
        # and the desired wheel position
        steering_motor_command = self._state.wheel_angle - action.command[1]

        self._state.set_pose(new_poses[0])
        self._state.wheel_angle = new_wheel_angles[0]
        self._state.steering_motor_command = steering_motor_command
        self._state.v = v[0]
        self._state.w = w[0]

    def draw(self, image, px, py, angle, color, map_resolution, alpha=1.0, draw_steering_details=True):
        """
        Draw robot on the image
        :param image: cv image to draw on
        :param px: pixel coordinates of the robot
        :param py: pixel coordinates of the robot
        :param angle: angle of the robot in drawing coordinates
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
                   (px + int(arrow_length * np.cos(angle)), py + int(arrow_length * np.sin(angle))),
                   color=(0, 0, 255),
                   thickness=2)

        wheel_pixel_x = int(px + np.cos(angle) * self.get_front_wheel_from_axis_distance() / map_resolution)
        wheel_pixel_y = int(py + np.sin(angle) * self.get_front_wheel_from_axis_distance() / map_resolution)

        if draw_steering_details:
            draw_arrow(image, (wheel_pixel_x, wheel_pixel_y),
                       (wheel_pixel_x + int(arrow_length * np.cos(angle - self._state.wheel_angle)),
                        wheel_pixel_y + int(arrow_length * np.sin(angle - self._state.wheel_angle))),
                       color=(0, 0, 255),
                       thickness=1)

        scrubdeck_status_str = ''
        for k, v in self._scrubdeck_status.items():
            if v:
                scrubdeck_status_str += '%s+' % k
        if len(scrubdeck_status_str) > 0:
            scrubdeck_status_str = scrubdeck_status_str[:-1]
            puttext_centered(image, scrubdeck_status_str, (image.shape[1] // 2, 10), color=(0, 0, 255))

    def set_noise_parameters(self, noise_parameters):
        """ Sets the noise parameters
        :param noise_parameters:  the noise parameters, TODO write me better
        """
        self._noise_parameters = noise_parameters

    def get_robot_type_name(self):
        return self._dimensions.get_name()

    def get_scrubdeck_status(self):
        """
        Get scrubdeck state
        :return Dict[String, object]: different fields for scrubber deck
        """
        return self._scrubdeck_status

    def apply_scrubdeck_control(self, scrubdeck_control):
        """
        Set scrubdeck state
        :param scrubdeck_control Dict[String, object]: different fields for scrubber deck
        """

        if len(scrubdeck_control) == 0:
            return

        if 'vac' in scrubdeck_control:
            self._scrubdeck_status['vacuum'] = bool(scrubdeck_control['vac'])
        if 'one_touch' in scrubdeck_control:
            self._scrubdeck_status['brushes'] = bool(scrubdeck_control['one_touch'])
