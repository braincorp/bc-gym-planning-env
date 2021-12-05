""" Drawing utilities """
from __future__ import absolute_import

from bc_gym_planning_env.utilities.path_tools import inscribed_radius
from bc_gym_planning_env.utilities.map_drawing_utils import get_drawing_coordinates_from_physical, \
    get_drawing_angle_from_physical, draw_world_map, draw_wide_path, prepare_canvas, draw_trajectory


def draw_robot(robot, image, x, y, angle, color, costmap, alpha=1.0, draw_steering_details=True):
    """
    Draw robot on the image
    :param robot IRobot: the robot that will supply the draw function
    :param image: cv image to draw on
    :param x: pixel coordinates of the robot
    :param y: pixel coordinates of the robot
    :param angle: angle of the robot in drawing coordinates
    :param color: color to draw with
    :param costmap: Costmap to draw on
    :param alpha float: transparency of the robot image
    :param draw_steering_details bool: Should draw state of steering on the image
    """
    px, py = get_drawing_coordinates_from_physical(costmap.get_data().shape,
                                                   costmap.get_resolution(),
                                                   costmap.get_origin(),
                                                   (x, y))
    pangle = get_drawing_angle_from_physical(angle)
    robot.draw(image, px, py, pangle, color, costmap.get_resolution(),
               alpha=alpha, draw_steering_details=draw_steering_details)


def draw_environment(path_to_follow, robot, costmap):
    """
    Draw obstacles, path and a robot
    :param path_to_follow: numpy array of (x, y, angle) of a path left to follow
    :param original_path: numpy array of (x, y, angle) of the original path that perhaps has been
                          followed up to some point.
    :param robot IRobot: the robot that will supply the draw function
    :param costmap Costmap: the costmap that will provide the obstacle data
    :return: numpy BGR rendered image
    """
    img = prepare_canvas(costmap.get_data().shape)

    draw_world_map(img, costmap.get_data())

    draw_trajectory(
        img,
        costmap.get_resolution(),
        costmap.get_origin(),
        path_to_follow,
        thickness=1
    )

    x, y, angle = robot.get_pose()
    draw_robot(robot, img, x, y, angle, color=(0, 100, 0), costmap=costmap)

    return img
