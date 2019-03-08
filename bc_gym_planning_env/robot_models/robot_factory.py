from __future__ import print_function
from __future__ import absolute_import
# ============================================================================
# Copyright 2015 BRAIN Corporation. All rights reserved. This software is
# provided to you under BRAIN Corporation's Beta License Agreement and
# your use of the software is governed by the terms of that Beta License
# Agreement, found at http://www.braincorporation.com/betalicense.
# ============================================================================
from bc_gym_planning_env.robot_models.robot_names import RobotNames

from bc_gym_planning_env.robot_models.differential_drive import DiffDriveRobot
from bc_gym_planning_env.robot_models.tricycle_model import TricycleRobot


def create_robot(robot_name, footprint_scale=1., front_wheel_angle=0.0):
    """
    Given a robot name (along with construction parameters common to all robots), create a new robot.

    :param robot_name: A string robot name: One of the RobotNames enum
    :param footprint_scale: Factor by which to scale the size of the footprint
    :param front_wheel_angle: Angle of the front wheel (only used when the robot is a tricycle
    :return: An IRobot object
    """
    if robot_name in (RobotNames.INDUSTRIAL_TRICYCLE_V1):
        robot = TricycleRobot(robots_type_name=robot_name, footprint_scale=footprint_scale, wheel_angle=front_wheel_angle,
                              dynamic_model=False)
        return robot
    elif robot_name == RobotNames.INDUSTRIAL_DIFFDRIVE_V1:
        return DiffDriveRobot(robots_type_name=robot_name, footprint_scale=footprint_scale)
    else:
        raise Exception('No robot named "{}" exists'.format(robot_name))
