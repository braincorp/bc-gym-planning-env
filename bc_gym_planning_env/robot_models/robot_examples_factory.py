"""Functions to easy create standard robots"""
from __future__ import print_function
from __future__ import absolute_import

from bc_gym_planning_env.robot_models.robot_dimensions_examples import get_dimensions_example
from bc_gym_planning_env.robot_models.standard_robot_names_examples import StandardRobotExamples

from bc_gym_planning_env.robot_models.differential_drive import DiffDriveRobot
from bc_gym_planning_env.robot_models.tricycle_model import TricycleRobot


def create_standard_robot(robot_name, footprint_scale=1., **kwargs):
    """
    Given a robot name (along with construction parameters common to all robots), create a new robot.

    :param robot_name: A string robot name: One of the RobotNames enum
    :param footprint_scale: Factor by which to scale the size of the footprint
    :param kwargs: Angle of the front wheel (only used when the robot is a tricycle
    :return: An IRobot object
    """
    dimensions = get_dimensions_example(robot_name)
    if robot_name in StandardRobotExamples.INDUSTRIAL_TRICYCLE_V1:
        robot = TricycleRobot(dimensions=dimensions, footprint_scale=footprint_scale, **kwargs)
        return robot
    elif robot_name == StandardRobotExamples.INDUSTRIAL_DIFFDRIVE_V1:
        return DiffDriveRobot(dimensions=dimensions, footprint_scale=footprint_scale, **kwargs)
    else:
        raise Exception('No robot named "{}" exists'.format(robot_name))
