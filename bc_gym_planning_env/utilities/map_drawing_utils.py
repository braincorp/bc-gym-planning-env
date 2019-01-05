from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np
from bc_gym_planning_env.utilities.costmap_2d_python import CostMap2D
from bc_gym_planning_env.utilities.path_tools import world_to_pixel, pixel_to_world, get_pixel_footprint, blit, draw_arrow


"""
NOTE: All draw functions in here assument that the image is already flipped for drawing... i.e. The lowet y value 
corresponds to the last row of the image array.
"""


def get_drawing_coordinates_from_physical(map_shape, resolution, origin, physical_coords, enforce_bounds=False):
    '''
    :param physical_coords: either (x, y)  or n x 2 array of (x, y), in physical units
    :param enforce_bounds: Can be:
        False: Allow points to be outside range of costmap
        True: Raise an error if points fall out of costmap
        'filter': Filter out points which fall out of costmap.
    :return: same in coordinates suitable for drawing (y axis is flipped)
    '''
    assert enforce_bounds in (True, False, 'filter')
    physical_coords = np.array(physical_coords)
    assert physical_coords.ndim <= 2
    assert physical_coords.shape[physical_coords.ndim - 1] == 2
    assert np.array(map_shape).ndim == 1

    pixel_coords = world_to_pixel(physical_coords, origin, resolution)
    # flip the y because we flip image for display
    pixel_coords[..., 1] = map_shape[0] - 1 - pixel_coords[..., 1]

    if enforce_bounds and (not (pixel_coords < map_shape[1::-1]).all() or (np.amin(pixel_coords) < 0)):
        raise IndexError("Point %s, in pixels (%s) is outside the map (shape %s)." % (physical_coords, pixel_coords, map_shape))
    return pixel_coords


def get_drawing_angle_from_physical(angle):
    '''
    Invert physical angle for consistency with inverting the y axis in
    get_drawing_coordinates_from_physical.
    :param angle: physical angle in radians
    :return: angle in radians to draw with
    '''
    return -angle


def get_physical_coords_from_drawing(map_shape, resolution, origin, drawing_coords):
    '''
    Inverse of the get_drawing_coordinates_from_physical function
    '''
    # this makes a copy to make sure that we do not change original coords
    drawing_coords = np.array(drawing_coords)
    assert drawing_coords.ndim <= 2
    assert drawing_coords.shape[drawing_coords.ndim - 1] == 2
    assert np.array(map_shape).ndim == 1
    drawing_coords[..., 1] = map_shape[0] - 1 - drawing_coords[..., 1]
    return pixel_to_world(drawing_coords, origin, resolution)


def get_physical_angle_from_drawing(angle):
    '''
    Invert drawing angle for consistency with inverting the y axis in
    get_physical_coords_from_drawing.
    :param angle: physical angle in radians in drawing coordinates
    :return: angle in radians to draw with
    '''
    return -angle


def get_pixel_footprint_for_drawing(angle, robot_footprint, map_resolution, fill=True):
    '''
    Return pixel footprint kernel for visualization of the robot.
    The footprint kernel is flipped.
    angle_range - angle in physical coordinates (!)
    '''
    footprint_picture = get_pixel_footprint(angle,
                                            robot_footprint, map_resolution, fill)
    footprint_picture = np.flipud(footprint_picture)
    return footprint_picture


def draw_trajectory(array_to_draw, resolution, origin, trajectory, color=(0, 255, 0),
                    enforce_bounds=False, with_orientation=False, thickness=1):
    if len(trajectory) == 0:
        return
    drawing_coords = get_drawing_coordinates_from_physical(
        array_to_draw.shape,
        resolution,
        origin,
        trajectory[:, :2],
        enforce_bounds=enforce_bounds)

    cv2.polylines(array_to_draw, [drawing_coords], False, color, thickness=thickness)
    if with_orientation:
        index = len(drawing_coords) - 2
        while (index >= 0 and np.array_equal(drawing_coords[index], drawing_coords[-1])):
            index -= 1
        if index >= 0:
            draw_arrow(array_to_draw, tuple(drawing_coords[index]), tuple(drawing_coords[-1] - drawing_coords[index]),
                       10, (255, 255, 255))


def _mark_wall_on_static_map(static_map, p0, p1, width, color):
    thickness = max(1, int(width/static_map.get_resolution()))
    cv2.line(
        static_map.get_data(),
        tuple(world_to_pixel(np.array(p0), static_map.get_origin(), static_map.get_resolution())),
        tuple(world_to_pixel(np.array(p1), static_map.get_origin(), static_map.get_resolution())),
        color=color,
        thickness=thickness)


def add_wall_to_static_map(static_map, p0, p1, width=0.05, cost=CostMap2D.LETHAL_OBSTACLE):
    _mark_wall_on_static_map(static_map, p0, p1, width, cost)


def remove_wall_from_static_map(static_map, p0, p1, width=0.05):
    _mark_wall_on_static_map(static_map, p0, p1, width, CostMap2D.FREE_SPACE)


def prepare_canvas(shape):
    """
    Prepare canvas for drawing
    :param shape (W, H): shape of the canvas
    :return array(W, H, 3)[uint8]: BGR canvas for drawing
    """
    return np.full(shape + (3,), 255, dtype=np.uint8)


def draw_world_map(img, costmap_data):
    '''
    Draws obstacles and unknowns
    :param img array(W, H, 3)[uint8]: canvas to draw on
    :param costmap_data(W, H)[uint8]: costmap data
    '''
    # flip image to show it in physical orientation like rviz
    costmap = np.flipud(costmap_data)
    img[costmap == CostMap2D.LETHAL_OBSTACLE] = (70, 70, 70)
    img[costmap == CostMap2D.NO_INFORMATION] = (20, 20, 20)


def draw_wide_path(img, path, robot_width, origin, resolution, color=(220, 220, 220)):
    """
    Draw a path as a tube to follow
    :param img array(N, M, 3)[uint8]: BGR image on which to draw (mutates image)
    :param path array(K, 3)[float]: array of (x, y, angle) of the path
    :param robot_width float: robot's width in meters
    :param origin array(2)[float]: x, y origin of the image
    :param resolution float: resolution of the costmap in meters
    :param color tuple[int]: BGR color tuple
    """
    drawing_coords = get_drawing_coordinates_from_physical(
        img.shape,
        resolution,
        origin,
        path[:, :2],
        enforce_bounds=False)

    cv2.polylines(img, [drawing_coords], False, color, thickness=int(robot_width / resolution))


def draw_robot(image_to_draw, footprint, pose, resolution, origin, color=(30, 150, 30), color_axis=None, fill=True):
    px, py = get_drawing_coordinates_from_physical(image_to_draw.shape,
                                                   resolution,
                                                   origin,
                                                   pose[0:2])
    kernel = get_pixel_footprint_for_drawing(pose[2], footprint, resolution, fill=fill)
    blit(kernel, image_to_draw, px, py, color, axis=color_axis)
    return px, py


def puttext_centered(im, text, pos, font=cv2.FONT_HERSHEY_PLAIN, size=0.6, color=(255, 255, 255)):
    text_size, _ = cv2.getTextSize(text, font, size, 1)
    y = int(pos[1] + text_size[1] // 2)
    x = int(pos[0] - text_size[0] // 2)  # it is complaining (integer argument expected)

    cv2.putText(im, text, (x, y), font, size, color)
