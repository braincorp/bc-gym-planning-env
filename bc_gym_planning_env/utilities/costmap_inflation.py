from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import cv2

from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.utilities.path_tools import circumscribed_radius, inscribed_radius

from bc_gym_planning_env.utilities.opencv_utils import single_threaded_opencv

INSCRIBED_INFLATED_OBSTACLE = 253


def distance_transform(img):
    '''
    Computes distance transform https://docs.opencv.org/3.4/d2/dbd/tutorial_distance_transform.html
    (Calculates the distance to the closest zero pixel for each pixel of the source image.)
    if walls are marked with 0, it will compute distance to the closes wall for each pixel.
    This function computes L2 precise distance (the most common usecase)
    :param np.array(W, H)[uint8] img: source image
    :return np.array(W, H)[float]: image with distances to 0 pixels
    '''
    with single_threaded_opencv():
        return cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def compute_cost_possibly_circumscribed_thresh(footprint, resolution, cost_scaling_factor):
    '''
    compute costs of possibly circumscribed radius needed for some planners (e.g. sbpl)
    :param footprint: n x 2 numpy array with ROS-style footprint (x, y coordinates)
    :param resolution: costmap resolution
    :param cost_scaling_factor: 1 over the distance (in world units) at which costs
            beyond the inscribed radius decay by a factor of e.
    '''
    distance = circumscribed_radius(footprint)
    inscribed_rad = inscribed_radius(footprint)
    distance_to_cost = _pixel_distance_to_cost(distance / resolution + 1, resolution,
                                               inscribed_rad, cost_scaling_factor)
    return max(0, distance_to_cost - 1)


def _pixel_distance_to_cost(distance, resolution, inscribed_rad, cost_scaling_factor):
    '''
    Transform a matrix or value of distance to obstacles (in pixels) into a matrix or value of costs
    '''
    if isinstance(distance, (float, int)):  # this isn't necessary with numpy 1.9.2
        distance = np.array([distance])
        scalar = True
    else:
        scalar = False
    pixel_inscribed_radius = inscribed_rad / resolution
    pixel_scaling_factor = cost_scaling_factor * resolution

    lethal = distance < (pixel_inscribed_radius/1000.)  # 1000 is probably a bug - the intent here that lethal is always false
    inscribed = distance <= pixel_inscribed_radius
    other = distance > pixel_inscribed_radius

    # ros conventions
    inscribed_cost = INSCRIBED_INFLATED_OBSTACLE
    lethal_cost = CostMap2D.LETHAL_OBSTACLE
    costs = np.zeros(distance.shape, dtype=np.uint8)
    costs[other] = (inscribed_cost - 1) * np.exp(-pixel_scaling_factor * (distance[other] - pixel_inscribed_radius))
    costs[inscribed] = inscribed_cost
    costs[lethal] = lethal_cost
    if scalar:
        return costs[0]
    else:
        return costs


def inflate_costmap(costmap, cost_scaling_factor, footprint):
    '''
    Inflate data with costs
    Costmap inflation is explained here:
    http://wiki.ros.org/costmap_2d#Inflation
    :param costmap CostMap2D: obstacle map
    :param cost_scaling_factor Float: how to scale the inflated costs
    :param footprint np.ndarray(N, 2)[Float]: footprint of the robot
    :return CostMap2D: inflated costs
    '''
    image_for_dist_transform = CostMap2D.LETHAL_OBSTACLE - costmap.get_data()
    distance = distance_transform(image_for_dist_transform)
    inscribed_rad = inscribed_radius(footprint)
    inflated_data = _pixel_distance_to_cost(distance, costmap.get_resolution(), inscribed_rad, cost_scaling_factor)

    return CostMap2D(
        data=inflated_data,
        origin=costmap.get_origin(),
        resolution=costmap.get_resolution()
    )
