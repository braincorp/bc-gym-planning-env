from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import cv2

from bc_gym_planning_env.utilities.coordinate_transformations import world_to_pixel
from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.utilities.opencv_utils import single_threaded_opencv
from bc_gym_planning_env.utilities.path_tools import get_pixel_footprint, get_blit_values


def extract_egocentric_costmap(costmap_2d, ego_position_in_world,
                               resulting_origin=None,
                               resulting_size=None, border_value=0):
    '''
    Returns a costmap as seen by robot at ego_position_in_world.
    In this costmap robot is at (0, 0) with 0 angle.
    :param ego_position_in_world - robot's position in the world costmap
    :param resulting_origin, resulting_size - Perform additional shifts and cuts so that
        resulting costmap origin and world size are equal to those parameters
    '''
    ego_pose = np.asarray(ego_position_in_world)
    pixel_origin = costmap_2d.world_to_pixel(ego_pose[:2])

    transform = cv2.getRotationMatrix2D(tuple(pixel_origin), 180 * ego_pose[2] / np.pi, scale=1)

    if resulting_size is None:
        resulting_size = costmap_2d.get_data().shape[:2][::-1]
    else:
        resulting_size = tuple(world_to_pixel(resulting_size, (0, 0), costmap_2d.get_resolution()))

    if resulting_origin is not None:
        resulting_origin = np.asarray(resulting_origin)
        assert (resulting_origin.shape[0] == 2)
        delta_shift = resulting_origin - (costmap_2d.get_origin() - ego_pose[:2])
        delta_shift_pixel = tuple(world_to_pixel(delta_shift, (0, 0), costmap_2d.get_resolution()))
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
    :param costmap: the 2d numpy array (the data of a costmap)
    :param angle: angle to rotate (in radians, world coordinates - positive angle is anticlockwise)
    :param border_value: value to fill in when rotating
    :param center_pixel_coords: center of rotation in the pixel coordinates (None for center of the image)
    :return: the rotated mat
    '''
    # opencv uses image coordintates, we use world coordinates
    deg_angle = np.rad2deg(-angle)
    if (deg_angle != 0.):
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


def is_obstacle(costmap, world_x, world_y, orientation=None, footprint=None):
    """
    Check costmap for obstacle at (world_x, world_y)
    If orientation is None, obstacle detection will check only the inscribed-radius
    distance collision. This means that, if the robot is not circular,
    there may be an undetected orientation-dependent collision.
    If orientation is given, footprint must also be given, and should be the same
    used by the costmap to inflate costs. Proper collision detection will
    then be done.
    """
    # convert to costmap coordinate system:
    map_x, map_y = costmap.world_to_pixel(np.array([world_x, world_y]))
    if not costmap.in_bounds(map_x, map_y):
        return False
    cost = costmap.get_data()[map_y, map_x]
    if (cost in [CostMap2D.LETHAL_OBSTACLE]):
        return True
    if orientation is None:  # only checking orientation-independent collision
        return False
    # Now check for orientation-dependent collision
    fp = get_pixel_footprint(orientation, footprint, costmap.get_resolution(), fill=True)
    values = get_blit_values(fp, costmap.get_data(), map_x, map_y)
    return np.any(values == CostMap2D.LETHAL_OBSTACLE)
