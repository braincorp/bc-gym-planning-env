""" Costmap utils, most importantly extracting egocentic view out of the costmap. """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import cv2

from bc_gym_planning_env.utilities.coordinate_transformations import world_to_pixel
from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.utilities.opencv_utils import single_threaded_opencv
from bc_gym_planning_env.utilities.path_tools import get_pixel_footprint, get_blit_mask, get_blit_values


def clone_costmap(costmap):
    '''
    Clone other costmap copying the data and other fields
    :param costmap CostMap2D: costmap to clone
    :return CostMap2D: costmap with copied data
    '''
    return CostMap2D(costmap.get_data().copy(),
                     costmap.get_resolution(),
                     costmap.get_origin().copy())


def extract_egocentric_costmap(costmap_2d, ego_position_in_world,
                               resulting_origin=None,
                               resulting_size=None, border_value=0):
    """
    Returns a costmap as seen by robot at ego_position_in_world.
    In this costmap robot is at (0, 0) with 0 angle.

    :param costmap_2d: global costmap
    :param ego_position_in_world: robot's position in the world costmap
    :param resulting_origin: Perform additional shifts and cuts so that
        resulting costmap origin and world size are equal to those parameters
    :param resulting_size: Perform additional shifts and cuts so that
        resulting costmap origin and world size are equal to those parameters
    :param border_value: the value at the border of the costmap to extrapolate
    :return: returns the egocentric costmap, with the robot in the middle.
    """
    ego_pose = np.asarray(ego_position_in_world)
    pixel_origin = costmap_2d.world_to_pixel(np.array(ego_pose[:2], dtype=np.float64))

    transform = cv2.getRotationMatrix2D(tuple(pixel_origin), 180 * ego_pose[2] / np.pi, scale=1)

    if resulting_size is None:
        resulting_size = costmap_2d.get_data().shape[:2][::-1]
    else:
        resulting_size = tuple(world_to_pixel(np.array(resulting_size, dtype=np.float64), np.array((0., 0.)), costmap_2d.get_resolution()))

    if resulting_origin is not None:
        resulting_origin = np.asarray(resulting_origin)
        assert resulting_origin.shape[0] == 2
        delta_shift = resulting_origin - (costmap_2d.get_origin() - ego_pose[:2])
        delta_shift_pixel = tuple(world_to_pixel(delta_shift, np.array((0., 0.)), costmap_2d.get_resolution()))
        shift_matrix = np.float32([[1, 0, -delta_shift_pixel[0]], [0, 1, -delta_shift_pixel[1]]])

        def _compose_affine_transforms(t1, t2):
            # http://stackoverflow.com/questions/13557066/built-in-function-to-combine-affine-transforms-in-opencv
            t1_expanded = np.array((t1[0], t1[1], (0, 0, 1)), dtype=np.float32)
            t2_expanded = np.array((t2[0], t2[1], (0, 0, 1)), dtype=np.float32)
            combined = np.dot(t2_expanded, t1_expanded)
            return combined[:2, :]

        transform = _compose_affine_transforms(transform, shift_matrix)
    else:
        resulting_origin = costmap_2d.get_origin() - ego_pose[:2]

    with single_threaded_opencv():
        rotated_data = cv2.warpAffine(costmap_2d.get_data(), transform, resulting_size,
                                      # this mode doesn't change the value of pixels during rotation
                                      flags=cv2.INTER_NEAREST, borderValue=border_value)

    return CostMap2D(rotated_data, costmap_2d.get_resolution(),
                     origin=resulting_origin)


def rotate_costmap(costmap, angle, center_pixel_coords=None, border_value=0):
    '''
    Rotate obstacle costmap around a particular center
    :param costmap: the 2d numpy array (the data of a costmap)
    :param angle: angle to rotate (in radians, world coordinates - positive angle is anticlockwise)
    :param center_pixel_coords: center of rotation in the pixel coordinates (None for center of the image)
    :param border_value: value to fill in when rotating
    :return: rotated costmap
    '''
    # opencv uses image coordintates, we use world coordinates
    deg_angle = np.rad2deg(-angle)
    if deg_angle != 0.:
        if center_pixel_coords is None:
            rows, cols = costmap.shape[:2]
            center_pixel_coords = (cols // 2, rows // 2)
        else:
            center_pixel_coords = tuple(center_pixel_coords)
        rot_mat = cv2.getRotationMatrix2D(center_pixel_coords, deg_angle, 1)
        with single_threaded_opencv():
            rotated_costmap = cv2.warpAffine(costmap, rot_mat, (costmap.shape[1], costmap.shape[0]),
                                             # this mode doesn't change the value of pixels during rotation
                                             flags=cv2.INTER_NEAREST,
                                             borderValue=border_value)
        return rotated_costmap
    else:
        return costmap.copy()


try:
    from brain.shining_utils.costmap_utils import is_footprint_colliding_impl

    def is_robot_colliding(robot_pose, footprint, costmap_data, origin, resolution):
        """
        Check costmap for obstacle at (world_x, world_y)
        If orientation is None, obstacle detection will check only the inscribed-radius
        distance collision. This means that, if the robot is not circular,
        there may be an undetected orientation-dependent collision.
        If orientation is given, footprint must also be given, and should be the same
        used by the costmap to inflate costs. Proper collision detection will
        then be done.
        :param robot_pose array(3)[float64]: pose of the robot
        :param footprint array(N, 2)[float64]: robot's footprint polygon
        :param costmap_data array(W, H)[uint8]: costmap obstacle data
        :param origin array(2)[float64]: origin (x, y) of the costmap
        :param resolution float: resolution of the costmap
        :return Bool: whether robot collides or not
        """
        # convert to costmap coordinate system:
        map_x, map_y = world_to_pixel(robot_pose[:2], origin, resolution)
        # TODO: remove this because the following is technically wrong: even if robot's origin is outside, it still can collide
        if not in_costmap_bounds(costmap_data, map_x, map_y):
            return False

        # Now check for orientation-dependent collision
        fp = get_pixel_footprint(robot_pose[2], footprint, resolution)
        image_slice, blit_mask = get_blit_mask(fp, costmap_data, map_x, map_y)
        if image_slice is None:
            return False
        return is_footprint_colliding_impl(image_slice, blit_mask, CostMap2D.LETHAL_OBSTACLE)

except ImportError:

    def is_robot_colliding(robot_pose, footprint, costmap_data, origin, resolution):
        """
        Check costmap for obstacle at (world_x, world_y)
        If orientation is None, obstacle detection will check only the inscribed-radius
        distance collision. This means that, if the robot is not circular,
        there may be an undetected orientation-dependent collision.
        If orientation is given, footprint must also be given, and should be the same
        used by the costmap to inflate costs. Proper collision detection will
        then be done.
        :param robot_pose array(3)[float64]: pose of the robot
        :param footprint array(N, 2)[float64]: robot's footprint polygon
        :param costmap_data array(W, H)[uint8]: costmap obstacle data
        :param origin array(2)[float64]: origin (x, y) of the costmap
        :param resolution float: resolution of the costmap
        :return Bool: whether robot collides or not
        """
        # convert to costmap coordinate system:
        map_x, map_y = world_to_pixel(robot_pose[:2], origin, resolution)
        # TODO: remove this because the following is technically wrong: even if robot's origin is outside, it still can collide
        if not in_costmap_bounds(costmap_data, map_x, map_y):
            return False
        # Now check for orientation-dependent collision
        fp = get_pixel_footprint(robot_pose[2], footprint, resolution)
        values = get_blit_values(fp, costmap_data, map_x, map_y)
        return np.any(values == CostMap2D.LETHAL_OBSTACLE)


def in_costmap_bounds(data, map_x, map_y):
    """
    whether a pixel at (map_x, map_y) is inside the costmap area
    :param data: data array
    :param map_x int: x coordinate
    :param map_y int: y coordinate
    :return bool: whether a pixel at (map_x, map_y) is inside the costmap area
    """
    return not (map_x < 0 or map_y < 0 or map_x >= data.shape[1] or map_y >= data.shape[0])


def pose_collides(pose, footprint, costmap_data, origin, resolution):
    """
    Check if robot footprint at x, y (world coordinates) and
        oriented as yaw collides with lethal obstacles.
    :param pose array(3)[float]: (x, y, angle) of the robot
    :param footprint array(N, 2)[float]: footprint polygon
    :param costmap_data array(W, H)[uint8]: costmap data with obstacles
    :param origin array(2)[float]: (x, y) origin of the costmap
    :param resolution float: resolution of the costmap
    :return bool: whether the robot collides or not
    """
    kernel_image = get_pixel_footprint(pose[2], footprint, resolution)
    # Get the coordinates of where the footprint is inside the kernel_image (on pixel coordinates)
    kernel = np.where(kernel_image)
    # Move footprint to (x,y), all in pixel coordinates
    x, y = world_to_pixel(pose[:2], origin, resolution)
    collisions = y + kernel[0] - kernel_image.shape[0] // 2, x + kernel[1] - kernel_image.shape[1] // 2

    # Check if the footprint pixel coordinates are valid, this is, if they are not negative and are inside the map
    good = np.logical_and(np.logical_and(collisions[0] >= 0, collisions[0] < costmap_data.shape[0]),
                          np.logical_and(collisions[1] >= 0, collisions[1] < costmap_data.shape[1]))

    # Just from the footprint coordinates that are good, check if they collide
    # with obstacles inside the map
    return bool(np.any(costmap_data[collisions[0][good],
                                    collisions[1][good]] == CostMap2D.LETHAL_OBSTACLE))
