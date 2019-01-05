from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
from bc_gym_planning_env.utilities.path_tools import world_to_pixel, pixel_to_world


CURRENT_ENCODING_VERSION = 1


class CostMap2D(object):
    """
    Helper class for working with CostMap
    """
    FREE_SPACE=0
    LETHAL_OBSTACLE=254
    NO_INFORMATION=255

    def __init__(self, data, resolution, origin):
        """
        :param data: A (width, height) numpy array
        :param resolution: A float, indicating the costmap resolution in meters/pixel.
        :param origin: A 2-element numpy array indicating the position, in meters, of the bottom-left corner of the costmap.
                      (i.e. the position corresponding to data[0, 0])
        """
        self._data = data
        self._resolution = resolution
        # origin has to be a numpy array with float type to avoid constant creation of arrays during perception
        assert origin.dtype == np.float64
        self._origin = freeze_array(origin)
        assert self._origin.dtype == np.float64
        assert not self._origin.flags.writeable

    @staticmethod
    def create_empty(world_size, resolution, world_origin=(0., 0.), dtype=np.uint8):
        """
        Create an empty costmap with a given size and resolution
        :param world_size: (x, y) size of world in meters.
        :param resolution: Resolution in meters/pixel
        :param world_origin: (x, y) location of the origin relative to the bottom left corner.
        :param dtype: Data type for costmap
        :return: A CostMap2D object
        """
        pixel_size = world_to_pixel(world_size, (0, 0), resolution)
        data = np.zeros(pixel_size[::-1], dtype=dtype)
        return CostMap2D(data, resolution, np.asarray(world_origin, dtype=np.float64))

    def get_resolution(self):
        return self._resolution

    def get_origin(self):
        assert not self._origin.flags.writeable
        return self._origin

    def __setstate__(self, state):
        self._data = state['_data']
        self._origin = freeze_array(state['_origin'])
        self._resolution = state['_resolution']

    def get_data(self):
        return self._data

    def in_bounds(self, map_x, map_y):
        """
        whether a pixel at (map_x, map_y) is inside the costmap area
        """
        return not (map_x < 0 or map_y < 0 or map_x >= self._data.shape[1] or map_y >= self._data.shape[0])

    def world_bounds(self):
        '''
        Return (min_x, max_x, min_y, max_y) of map in world coordinates
        '''
        x, y = self._origin
        return (x, x + self._resolution * self._data.shape[1], y, y + self._resolution * self._data.shape[0])

    def world_size(self):
        '''
        Return (size_x, size_y) of map in world coordinates
        '''
        xmin, xmax, ymin, ymax = self.world_bounds()
        return np.array([xmax-xmin, ymax-ymin])

    def world_center(self):
        '''
        Return (x, y) of map center in world coordinates
        '''
        xmin, xmax, ymin, ymax = self.world_bounds()
        return np.array([xmax+xmin, ymax+ymin])*0.5

    def world_to_pixel(self, world_coords):
        """
        Convert world coordinates to map pixel coordinates
        Note that this does not check whether the converted coordinate
        :param world_coords: either a n-elem numpy array [x, y] or a n x 2 array with n points
        :return: same as input, but in pixel coordinates.
        """
        return world_to_pixel(world_coords, self._origin, self._resolution)

    def pixel_to_world(self, pixel_coords):
        """
        Convert map pixel coordinates to world coordinates
        Note that this does not check whether the coordinates
        are in bounds.
        :param pixel_coords: either a n-elem numpy array [x, y] or a n x 2 array with n points
        :return: same as input, but in world coordinates
        """
        return pixel_to_world(pixel_coords, self._origin, self._resolution)

    def get_state(self):
        return dict(
            version=CURRENT_ENCODING_VERSION,
            data=self._data,
            resolution=self._resolution,
            origin=self._origin
        )

    @classmethod
    def from_state(cls, state):
        assert(state['version'] == CURRENT_ENCODING_VERSION)
        return CostMap2D(
            data=state['data'],
            resolution=state['resolution'],
            origin=freeze_array(state['origin']),
        )


def freeze_array(array):
    """
    Make numpy array read-only
    :param array: numpy array
    :return: read-only numpy array
    """
    array.flags.writeable = False
    return array
