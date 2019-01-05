from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import cv2

from bc_gym_planning_env.utilities.costmap_2d_python import CostMap2D
from bc_gym_planning_env.utilities.opencv_utils import single_threaded_opencv
from bc_gym_planning_env.utilities.path_tools import world_to_pixel


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
