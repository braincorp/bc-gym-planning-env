from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

from bc_gym_planning_env.utilities.map_drawing_utils import get_pixel_footprint_for_drawing, get_physical_angle_from_drawing, \
    puttext_centered
from bc_gym_planning_env.robot_models.standard_robot_names_examples import StandardRobotExamples
from bc_gym_planning_env.utilities.path_tools import blit, draw_arrow, path_velocity
from bc_gym_planning_env.robot_models.differential_drive import kinematic_body_pose_motion_step, \
    kinematic_body_pose_motion_step_with_noise
from bc_gym_planning_env.robot_models.interface import IRobot
from bc_gym_planning_env.robot_models.standard_robot_names_examples import RobotDriveTypes
from bc_gym_planning_env.utilities.robot_dimensions import get_dimensions_by_name


def vw_from_front_wheel_velocity(front_wheel_velocity, front_wheel_angle, front_wheel_from_axis):
    '''
    Compute linear and angular velocity of the back wheels center based on front wheel velocity
    and angle
    '''
    v = front_wheel_velocity * np.cos(front_wheel_angle)
    w = front_wheel_velocity * np.sin(front_wheel_angle) / front_wheel_from_axis
    return v, w


def tricycle_kinematic_step(pose, current_wheel_angle, dt, control_signals, max_front_wheel_angle, front_wheel_from_axis,
                            max_front_wheel_speed, front_column_p_gain, model_front_column_pid=True):
    '''
    :param pose: ... x 3  (x, y, angle)
    :param pose: ... x 1  (wheel_angle)
    :param dt: integration timestep
    :param control_signals: ... x 2 (wheel_v, wheel_angle) controls
    :return: ... x 3 pose and ... x 1 state (wheel_angle) after one timestep
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
    :param max_front_wheel_angle, front_wheel_from_axis, max_front_wheel_speed,
        max_linear_acceleration, max_angular_acceleration - parameters of the model
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
    '''
    The model of the front wheel column which includes PID and takes into account the constraints
    on the angular velocity of the wheel and maximum angle
    '''
    # rotate the front wheel first emulating a pid controller on the front wheel with a finite rotation speed
    max_front_wheel_delta = max_front_wheel_speed*dt
    clip_first = False
    if clip_first:
        desired_wheel_delta = desired_front_wheel_angle - current_front_wheel_angle
        desired_wheel_delta = np.clip(desired_wheel_delta, -max_front_wheel_delta, max_front_wheel_delta)
        new_front_wheel_angle = current_front_wheel_angle + front_column_p_gain*desired_wheel_delta
    else:
        desired_wheel_delta = front_column_p_gain*(desired_front_wheel_angle - current_front_wheel_angle)
        desired_wheel_delta = np.clip(desired_wheel_delta, -max_front_wheel_delta, max_front_wheel_delta)
        new_front_wheel_angle = current_front_wheel_angle + desired_wheel_delta
    new_front_wheel_angle = np.clip(new_front_wheel_angle, -max_front_wheel_angle, max_front_wheel_angle)
    return new_front_wheel_angle


def tricycle_velocity_dynamic_model_step(
        current_v, current_w, current_wheel_angle, desired_wheel_v,
        front_wheel_from_axis, max_linear_acceleration, max_angular_acceleration, dt):

    desired_linear_velocity = desired_wheel_v * np.cos(current_wheel_angle)
    desired_angular_velocity = desired_wheel_v * np.sin(current_wheel_angle) / front_wheel_from_axis

    desired_linear_acceleration = (desired_linear_velocity - current_v)/dt
    desired_angular_acceleration = (desired_angular_velocity - current_w)/dt

    # we can decelerate twice as fast as accelerate
    linear_acceleration = np.clip(desired_linear_acceleration, -2*max_linear_acceleration, max_linear_acceleration)
    angular_acceleration = np.clip(desired_angular_acceleration, -max_angular_acceleration, max_angular_acceleration)

    new_linear_velocity = (current_v + linear_acceleration*dt)
    new_angular_velocity = (current_w + angular_acceleration*dt)

    return new_linear_velocity, new_angular_velocity


class TricycleRobot(IRobot):

    def __init__(self, center_shift=0., footprint_scale=1.0, robots_type_name=StandardRobotExamples.INDUSTRIAL_TRICYCLE_V1,
                 wheel_angle=0.,
                 model_front_column_pid=True,
                 dynamic_model=True
                 ):
        '''
        center_shift: how much the center of the robot is displaced from the back axis.
            Helpful, for example, if we want to follow a path defined by the center of
            the robot and not the point between the wheels (for this, set it to 0.45)
        footprint_scale: a factor to make the footprint smaller or bigger than the actual
        robots_type_name: A string with the robot's type name, which should be the RobotNames enum
        :param model_front_column_pid bool: turn on PID modeling or assume immediate steering wheel rotation
        :param dynamic_model bool: use dynamic model with mass or a kinematic one
        '''
        self._x = 0.
        self._y = 0.
        self._measured_v = 0.
        self._measured_w = 0.
        self._steering_motor_command = 0.
        self._model_front_column_pid = model_front_column_pid
        self._dynamic_model = dynamic_model

        self._angle = 0.
        self._last_pose = (self._x, self._y, self._angle)

        self._center_shift = center_shift
        self._footprint_scaling = footprint_scale

        self._wheel_angle = wheel_angle
        self._noise_parameters = None

        self._scrubdeck_status = {
            'vacuum': False,
            'brushes': False
        }

        self._dimensions = get_dimensions_by_name(robots_type_name)

    def enable_model_front_column_pid(self, enable):
        self._model_front_column_pid = enable

    def get_drive_type(self):
        return RobotDriveTypes.TRICYCLE

    def get_footprint(self):
        footprint = self._dimensions.footprint()*self._footprint_scaling
        footprint[:, 0] -= self._center_shift
        return footprint

    def get_footprint_scale(self):
        return self._footprint_scaling

    def get_pose(self):
        return np.array([
            self._x + self._center_shift*np.cos(self._angle),
            self._y + self._center_shift*np.sin(self._angle),
            self._angle
        ])

    def get_front_wheel_angle(self):
        return self._wheel_angle

    def get_robot_state(self):
        return [self._wheel_angle, self._measured_v, self._measured_w, self._steering_motor_command]

    def set_front_wheel_angle(self, angle):
        self._wheel_angle = angle

    def get_dimensions(self):
        return self._dimensions

    def get_front_wheel_from_axis_distance(self):
        return self._dimensions.front_wheel_from_axis()

    def get_max_front_wheel_angle(self):
        return self._dimensions.max_front_wheel_angle()

    def get_max_front_wheel_speed(self):
        return self._dimensions.max_front_wheel_speed()

    def get_max_linear_acceleration(self):
        return self._dimensions.max_linear_acceleration()

    def get_max_angular_acceleration(self):
        return self._dimensions.max_angular_acceleration()

    def get_front_column_model_p_gain(self):
        return self._dimensions.front_column_model_p_gain()

    def get_default_controls(self):
        return 0., self.get_front_wheel_angle()

    def set_pose(self, x, y, angle):
        self._x = x - self._center_shift*np.cos(angle)
        self._y = y - self._center_shift*np.sin(angle)
        self._angle = angle
        self._last_pose = (self._x, self._y, self._angle)
        self._measured_v = 0.
        self._measured_w = 0.

    def step(self, dt, control_signals):
        '''
        step when control signals are wheel linear speed and wheel angle
        '''
        if self._dynamic_model:
            new_poses, new_wheel_angles, _, _ =\
                tricycle_dynamic_step(
                    np.array([[self._x, self._y, self._angle]]),
                    [self._wheel_angle], [self._measured_v], [self._measured_w],
                    dt, np.array([control_signals]),
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
            new_poses, new_wheel_angles =\
                tricycle_kinematic_step(
                    np.array([[self._x, self._y, self._angle]]),
                    [self._wheel_angle],
                    dt, np.array([control_signals]),
                    self.get_max_front_wheel_angle(),
                    self.get_front_wheel_from_axis_distance(),
                    self.get_max_front_wheel_speed(),
                    self.get_front_column_model_p_gain(),
                    model_front_column_pid=self._model_front_column_pid
                )

        (self._x, self._y, self._angle), self._wheel_angle = new_poses[0], new_wheel_angles[0]
        v, w = path_velocity([(0,) + self._last_pose,
                              (dt,) + (self._x, self._y, self._angle)])
        self._measured_v = v[0]
        self._measured_w = w[0]

        # This is a very crude estimation of the steering motor command angle based on the current angle of the wheel
        # and the desired wheel position
        self._steering_motor_command = self._wheel_angle - control_signals[1]

        self._last_pose = (self._x, self._y, self._angle)

    def draw(self, image, px, py, angle, color, map_resolution):
        '''
        Draw robot on the image
        :param image: cv image to draw on
        :param px, py: pixel coordinates of the robot
        :param angle: angle of the robot in drawing coordinates
        :param color: color to draw with
        :param map_resolution: resolution of the image
        '''

        # get_pixel_footprint_for_drawing takes angle in physical coordinates
        kernel = get_pixel_footprint_for_drawing(get_physical_angle_from_drawing(angle),
                                                 self.get_footprint(),
                                                 map_resolution)
        blit(kernel, image, px, py, color)

        px -= int(self._center_shift*np.cos(angle)/map_resolution)
        py -= int(self._center_shift*np.sin(angle)/map_resolution)

        arrow_length = 15
        draw_arrow(image, (px, py),
                   (px+int(arrow_length*np.cos(angle)), py+int(arrow_length*np.sin(angle))),
                   color=(0, 0, 255),
                   thickness=2)

        wheel_pixel_x = int(px + np.cos(angle)*self.get_front_wheel_from_axis_distance()/map_resolution)
        wheel_pixel_y = int(py + np.sin(angle)*self.get_front_wheel_from_axis_distance()/map_resolution)

        draw_arrow(image, (wheel_pixel_x, wheel_pixel_y),
                   (wheel_pixel_x+int(arrow_length*np.cos(angle-self._wheel_angle)),
                    wheel_pixel_y+int(arrow_length*np.sin(angle-self._wheel_angle))),
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
        self._noise_parameters = noise_parameters

    def get_scrubdeck_status(self):
        return self._scrubdeck_status

    def apply_scrubdeck_control(self, scrubdeck_control):
        if len(scrubdeck_control) == 0:
            return

        if 'vac' in scrubdeck_control:
            self._scrubdeck_status['vacuum'] = bool(scrubdeck_control['vac'])
        if 'one_touch' in scrubdeck_control:
            self._scrubdeck_status['brushes'] = bool(scrubdeck_control['one_touch'])
