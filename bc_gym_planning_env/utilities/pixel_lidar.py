from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
# ============================================================================
# Copyright 2015 BRAIN Corporation. All rights reserved. This software is
# provided to you under BRAIN Corporation's Beta License Agreement and
# your use of the software is governed by the terms of that Beta License
# Agreement, found at http://www.braincorporation.com/betalicense.
# ============================================================================

import numpy as np

from bc_gym_planning_env.utilities.coordinate_transformations import normalize_angle
from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.utilities.scan_tools import scan_to_cartesian
from bc_gym_planning_env import PixelRaytraceWrapper, raytrace_clean_on_input_map


class PixelLidar(object):
    '''
    Given a numpy map it returns the (x, y) points in the map
    that would be detected by a lidar in a given position
    and orientation
    '''
    def __init__(self, pixel_radius, angle_range=None, blind_radius=0,
                 n_angles=None):
        '''
        pixel_radius: how far the lidar reaches, in pixels
        angle_range: the angle range of the lidar, in radians (if None, 2*pi)
        blind_radius: points closer than this (in pixels) won't
            be detected
        n_angles: defines the angular resolution; it is the number of angles
            *in a whole circumference*; if the angle_range is not a whole circumference,
            the number of values you will get when calling get_scan_ranges
            will not be n_angles but the proportional part.
            If None, the number of angles used will be 1.5 times the
            pixel circumference at pixel_radius.
            Also notice that, because the rays are precalculated, the absolute
            directions of the rays are fixed, starting at 0 angle,
            and when a scan is required at a given orientation the orientation
            is rounded to these fixed directions. E.g., if n_angles is 4,
            the scans will be at fixed world orientations 0, 90, 180 and 270
            degrees, independent of robot orientation.
        '''
        assert pixel_radius > 0, "Minimum pixel radius is 1"

        if n_angles is None:
            # take 50% more than the number of pixels in the pixel_radius circumference
            n_angles = int(np.ceil(3. * np.pi * pixel_radius))
        assert n_angles > 0, "Need at least one ray"
        self._angles = np.linspace(0, 2 * np.pi, 1 + n_angles)[:-1]

        self._d_angle = np.diff(self._angles)[0]
        self._half_range_idx = None if angle_range is None else np.round(angle_range / 2. / self._d_angle).astype(np.int)

        # self._x and self._y have one row per angle and contain the 0-centered
        # coordinates of each tracing ray
        start_x = np.round(blind_radius * np.cos(self._angles)).astype(np.int16)
        start_y = np.round(blind_radius * np.sin(self._angles)).astype(np.int16)
        end_x = np.round(pixel_radius * np.cos(self._angles)).astype(np.int16)
        end_y = np.round(pixel_radius * np.sin(self._angles)).astype(np.int16)
        self._line_defs = np.ascontiguousarray(np.vstack((start_x, start_y, end_x, end_y)).T)
        self._pixel_radius = pixel_radius
        self._raytrace_module = PixelRaytraceWrapper()

    def get_scan_points(self, im, center, orientation):
        '''
        im: 2d uint8 image with 254 for obstacles and anything else for empty space
        center: (x, y), in pixels, in the image
        orientation: in radians

        Returns two n x 2 numpy array of (x, y) points, in pixel coordinates,
         - encountered by the lidar.
         - cleared by the lidar (Warning: coordinates might fall outside the image coordinates!)
        '''
        if self._half_range_idx is not None:
            or_idx = np.round(np.mod(orientation, 2 * np.pi) / self._d_angle).astype(np.int)
            idx = np.sort(np.mod(np.arange(or_idx - self._half_range_idx, or_idx + self._half_range_idx + 1), len(self._angles)))
        else:
            idx = np.arange(len(self._line_defs))
        detection = self._raytrace_module.pixel_lidar(im, self._line_defs, idx.astype(np.int32),
                                                      center[0], center[1], 254)
        hits = detection[:, 0] >= 0
        result = detection[hits]
        clearing = np.ascontiguousarray(self._line_defs[idx[~hits], 2:4] + np.array(center, dtype=np.int16))

        # Get unique results (http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array)
        unique_result = np.unique(result.view(np.dtype((np.void, result.dtype.itemsize * 2)))).view(result.dtype).reshape(-1, 2)
        unique_clearing = np.unique(clearing.view(np.dtype((np.void, clearing.dtype.itemsize * 2)))).view(clearing.dtype).reshape(-1, 2)
        return unique_result, unique_clearing

    def get_scan_ranges(self, im, center, orientation, lidar_angles=None):
        '''
        im: 2d uint8 image with 254 for obstacles and anything else for empty space
        center: (x, y), in pixels, in the image
        orientation: in radians
        lidar_angles: custom angles of a lidar (otherwise angles follow the lidar's range_angular and resolution_angular construction params)
        Returns:
            - 2 x n array:
                first row - array of the (ego) orientations (in radians) that correspond to each range
                second row - array of ranges (in pixels) centered at the orientation angle.

            - 2 x n array:
                first row - array of the (ego) orientations (in radians) that correspond to each range
                second row - array of clearing ranges centered at the orientation angle where rays didn't hit the target
                    The value of clearing range is pixel_radius
        '''

        if lidar_angles is not None:
            selected_angles = (np.mod(orientation + lidar_angles, 2 * np.pi) / self._d_angle + 0.5)
            selected_angles = np.mod(selected_angles, self._line_defs.shape[0]).astype(np.int32)
        else:
            or_idx = np.int(np.mod(orientation, 2 * np.pi) / self._d_angle + 0.5)
            if self._half_range_idx is not None:
                selected_angles = np.mod(np.arange(or_idx - self._half_range_idx, or_idx + self._half_range_idx + 1), len(self._angles))
            else:
                selected_angles = np.arange(len(self._angles))
                selected_angles = np.roll(selected_angles, len(self._angles) // 2 - or_idx)  # rotate the ranges to be centered at the orientation angle

        assert selected_angles.max() < self._line_defs.shape[0]

        detection = self._raytrace_module.pixel_lidar(im, self._line_defs, selected_angles.astype(np.int32),
                                                      center[0], center[1], 254)
        hits = detection[:, 0] >= 0
        result = detection[hits]
        r = np.zeros(len(selected_angles))
        r[hits] = np.hypot(result[:, 0] - center[0], result[:, 1] - center[1])
        r[~hits] = np.NAN

        if lidar_angles is not None:
            angles = lidar_angles
        else:
            angles = normalize_angle(self._angles[selected_angles] - orientation)
        range_scan = np.vstack((angles, r))
        clearing_scan = range_scan.copy()
        clearing_scan[1, :] = np.inf
        clearing_scan[1, ~hits] = self._pixel_radius
        return range_scan, clearing_scan

    def get_clear_space(self, im, center, orientation, lidar_angles=None):
        '''
        This function will raytracing on the current costmap and return all the clear space
        :param im: 2d uint8 image with 254 for obstacles and anything else for empty space
        :param center: (x, y), in pixels, in the image
        :param orientation: in radians
        :param lidar_angles: custom angles of a lidar (otherwise angles follow the lidar's range_angular and resolution_angular construction params)
        :return a map with the same size as im but filled with empty space
        '''

        part = np.zeros(im.shape, dtype=np.uint16)

        if lidar_angles is not None:
            selected_angles = (np.mod(orientation + lidar_angles, 2 * np.pi) / self._d_angle + 0.5)
            selected_angles = np.mod(selected_angles, self._line_defs.shape[0]).astype(np.int32)
        else:
            or_idx = np.int(np.mod(orientation, 2 * np.pi) / self._d_angle + 0.5)
            if self._half_range_idx is not None:
                selected_angles = np.mod(np.arange(or_idx - self._half_range_idx, or_idx + self._half_range_idx + 1), len(self._angles))
            else:
                selected_angles = np.arange(len(self._angles))
                selected_angles = np.roll(selected_angles, len(self._angles) // 2 - or_idx)  # rotate the ranges to be centered at the orientation angle

        assert selected_angles.max() < self._line_defs.shape[0]

        raytrace_clean_on_input_map(im, part, self._line_defs, selected_angles.astype(np.int32),
                                    center[0], center[1], 254)

        return part

    def get_angles(self):
        '''
        return array of the (ego) orientations (in radians) that correspond to each range at 0 orientation
        '''
        or_idx = np.int(np.mod(0, 2 * np.pi) / self._d_angle + 0.5)
        if self._half_range_idx is not None:
            selected_angles = np.mod(np.arange(or_idx - self._half_range_idx, or_idx + self._half_range_idx + 1), len(self._angles))
        else:
            selected_angles = np.arange(len(self._angles))
            selected_angles = np.roll(selected_angles, len(self._angles) // 2 - or_idx)  # rotate the ranges to be centered at the orientation angle
        angles = normalize_angle(self._angles[selected_angles])
        return angles


class VirtualLidar(object):
    '''
    Wrapper on top of PixelLidar working in world coordinates
    '''
    def __init__(self, range_max, range_angular, costmap, resolution_angular=None):
        '''
        :param range_max: max lidar range, in meters
        :param range_angular: in radians, None for 2pi lidar
        :param resolution_angular: in radians
        :param costmap: CostMap2D instance
        :return:
        '''
        assert isinstance(costmap, CostMap2D)
        n_angles = None
        if resolution_angular is not None:
            n_angles = int(2.0 * np.pi / resolution_angular + 0.5)
        self._pixel_lidar = PixelLidar(pixel_radius=int(range_max/costmap.get_resolution()),
                                       angle_range=range_angular,
                                       blind_radius=0,
                                       n_angles=n_angles)
        self._costmap = costmap
        self._range_max = range_max
        self._range_angular = range_angular
        self._resolution_angular = resolution_angular

    def get_scan_points(self, pose):
        '''
        :param pose: [x, y, angle], in world coordinates

        :return: two n x 2 numpy array of (x, y) points, in world coordinates,
         - encountered by the lidar.
         - cleared by the lidar (Warning: coordinates might fall outside the map coordinates!)
        '''
        im = self._costmap.get_data()
        pixel_location = self._costmap.world_to_pixel(pose[:2])
        pixel_scan, clearing_pixel_scan = self._pixel_lidar.get_scan_points(im, pixel_location, pose[2])
        world_scan = self._costmap.pixel_to_world(pixel_scan)
        world_clearing_scan = self._costmap.pixel_to_world(clearing_pixel_scan)
        return world_scan, world_clearing_scan

    def get_scan_points_egocentric(self, pose):
        '''
        :param pose: [x, y, angle], in world coordinates
        :return:  two n x 2 numpy array of (x, y) points, in pose coordinates,
         - encountered by the lidar.
         - cleared by the lidar (Warning: coordinates might fall outside the map coordinates!)
        '''

        ranges, clearing = self.get_scan_ranges(pose)
        return scan_to_cartesian(ranges[1, :], ranges[0, :]), scan_to_cartesian(clearing[1, :], clearing[0, :])

    def get_scan_ranges(self, pose, lidar_angles=None):
        '''
        :param pose: the pose of the lidar in the world as [x, y, angle]
        :param lidar_angles: custom angles of a lidar (otherwise angles follow the lidar's range_angular and resolution_angular construction params)

        :return:
            - 2 x n array:
                first row - array of the (ego) orientations (in radians) that correspond to each range
                second row - array of ranges (in pixels) centered at the orientation angle.

            - 2 x n array:
                first row - array of the (ego) orientations (in radians) that correspond to each range
                second row - array of clearing ranges centered at the orientation angle where rays didn't hit the target
                    The value of clearing range is pixel_radius
        '''
        im = self._costmap.get_data()
        pixel_location = self._costmap.world_to_pixel(pose[:2])
        pixel_range_scan, clearing_range_pixel_scan = self._pixel_lidar.get_scan_ranges(im, pixel_location, pose[2], lidar_angles=lidar_angles)
        pixel_range_scan[1, :] *= self._costmap.get_resolution()
        clearing_range_pixel_scan[1, :] *= self._costmap.get_resolution()
        return pixel_range_scan, clearing_range_pixel_scan

    def get_clear_space(self, pose, lidar_angles=None):
        '''
        This function will raytracing on the current costmap and return all the clear space
        :param pose: the pose of the lidar in the world as [x, y, angle]
        :param lidar_angles: custom angles of a lidar (otherwise angles follow the lidar's range_angular and resolution_angular construction params)
        :return: a map with the same size as im but filled with empty space
        '''

        im = self._costmap.get_data()
        pixel_location = self._costmap.world_to_pixel(pose[:2])
        part = self._pixel_lidar.get_clear_space(im, pixel_location, pose[2], lidar_angles=lidar_angles)

        return part

    def get_angles(self):
        return self._pixel_lidar.get_angles()

    def get_range_max(self):
        '''
        :return: the maximum range of the virtual lidar in meters
        '''
        return self._range_max

    def get_range_min(self):  # pylint: disable=no-self-use
        '''
        :return: the minimum range of the virtual lidar in meters
        '''
        return 0.

    def get_range_angular(self):
        '''
        :return: the angular range of the virtual lidar in radians
        '''
        return self._range_angular

    def get_resolution_angular(self):
        '''
        :return: the angular resolution of the virtual lidar in radians/sample
        '''
        return self._resolution_angular
