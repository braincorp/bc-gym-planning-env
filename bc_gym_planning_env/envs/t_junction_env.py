"""
Synthetic and its corresponding static path for T-Junction
"""

import attr
import numpy as np

from typing import Sequence

from bc_gym_planning_env.envs.base.maps import Wall
from bc_gym_planning_env.utilities.costmap_2d import CostMap2D


class Bearing:
    """
    Defining the direction robot would face given starting and ending positions
    """
    starting = {
        "top": 3*np.pi/2,
        "bottom": np.pi/2,
        "left": 0.0,
        "right": np.pi
    }
    ending = {
        "top": np.pi/2,
        "bottom": 3*np.pi/2,
        "left": np.pi,
        "right": 0.0
    }



@attr.s
class TJunction:
    """
    How the T-Junction is defined as well as parametrization of the environment
                            window_width
                  <---------------------------->
                {  E---------------------------D
                {                               } beam_width
                {  F---------G     B-----------C
 window_height  {            |     |
                {            |     |
                {            |     |
                {            |     |
                             O     A
                             <----->
                           column_width
    """
    start_noise_scale = attr.ib(type=float, default=0.0)  # Using SI units
    window_height = attr.ib(type=float, default=10.0)  # Using SI units
    window_width = attr.ib(type=float, default=10.0)  # Using SI units
    column_width = attr.ib(type=float, default=1.5)  # Using SI units
    beam_width = attr.ib(type=float, default=1.5)  # Using SI units
    mu = attr.ib(type=list, default=None)  # Gaussian mean
    sigma = attr.ib(type=list, default=None)  # Standard deviation
    wall_corners = attr.ib(type=np.array, default=None)
    obstacles = attr.ib(type=Sequence[Wall], default=None)

    _o = attr.ib(type=np.array, default=None)
    _a = attr.ib(type=np.array, default=None)
    _b = attr.ib(type=np.array, default=None)
    _c = attr.ib(type=np.array, default=None)
    _d = attr.ib(type=np.array, default=None)
    _e = attr.ib(type=np.array, default=None)
    _f = attr.ib(type=np.array, default=None)
    _g = attr.ib(type=np.array, default=None)

    _walls_defined = attr.ib(type=bool, default=False)

    def __attrs_post_init__(self):
        """
        init function after attributes have  been initialized
        :return:
        """
        self._check_input_arg_values()
        self.wall_corners = self.get_map_standard_coordinates()
        self.obstacles = self.get_map_walls()

    def _check_input_arg_values(self):
        """
        Ensure that arguments relating to the dimensional
        :return:
        """
        if not self.column_width > 0:
            raise ValueError("column_width_left must be a positive real number greater than 0")
        if not self.beam_width > 0:
            raise ValueError("beam_width must be a positive real number greater than 0")
        if not self.window_height > 0:
            raise ValueError("beam_width must be a positive real number greater than 0")
        if not self.window_width > 0:
            raise ValueError("window_width must be a positive real number greater than 0")
        if not self.column_width <= self.window_width:
            raise ValueError("column_width must be less than or equal to window_width")
        if not self.beam_width <= self.window_height:
            raise ValueError("beam_width must be less than equal to window_height")

    def get_map_standard_coordinates(self):
        """
        Determine the coordinates of the wall vertices
        :return: The coordinates of the end points and vertices of the map as defined above
        """

        x_origin = 0.0
        y_origin = 0.0

        # value used to determine the x of d,e,f,c
        alpha = (self.window_width - self.column_width) / 2.0
        self._o = np.array([x_origin, y_origin])
        self._a = np.array([x_origin + self.column_width, y_origin])
        self._b = np.array([x_origin + self.column_width, y_origin + self.window_height - self.beam_width])
        self._c = np.array([x_origin + self.column_width + alpha, y_origin + self.window_height - self.beam_width])
        self._d = np.array([x_origin + self.column_width + alpha, y_origin + self.window_height])
        self._e = np.array([x_origin - alpha, y_origin + self.window_height])
        self._f = np.array([x_origin - alpha, y_origin + self.window_height - self.beam_width])
        self._g = np.array([x_origin, y_origin + self.window_height - self.beam_width])

        self._walls_defined = True

        return np.array([self._o, self._a, self._b, self._c, self._d, self._e, self._f, self._g])

    def get_map_walls(self):
        """
        Create a list of wall objects
        :return Sequence[ICostmapRenderer]: obstacles, list of Wall objects
        """
        if not self._walls_defined:
            raise AttributeError("coordinates of the wall have not been defined")

        obstacles = [
            Wall(from_pt=self._a, to_pt=self._b),
            Wall(from_pt=self._b, to_pt=self._c),
            Wall(from_pt=self._d, to_pt=self._e),
            Wall(from_pt=self._f, to_pt=self._g),
            Wall(from_pt=self._g, to_pt=self._o)
        ]
        return obstacles

    def get_costmap(self, resolution=0.03):
        """
        gets the static map based on the generated coordinates of the synthetic map
        :param resolution: meters per pixel resolution
        :return: CostMap2D object
        """

        if not self._walls_defined:
            raise AttributeError("coordinates of the wall have not been defined")

        margin = 1.0  # margin of 1 meters around the walls

        # Index naming for readability
        x = 0
        y = 1

        min_x = np.min(self.wall_corners[:, x])
        min_y = np.min(self.wall_corners[:, y])
        max_x = np.max(self.wall_corners[:, x])
        max_y = np.max(self.wall_corners[:, y])
        world_size = abs(max_x - min_x) + 2 * margin, abs(max_y - min_y) + 2 * margin
        world_origin = min_x - margin, min_y - margin  # world origin is min_x and min_y of the synthetic map

        static_map = CostMap2D.create_empty(
            world_size=world_size,
            world_origin=world_origin,
            resolution=resolution
        )

        for renderer in self.obstacles:
            renderer.render(static_map)

        return static_map

    def get_path(self, starting_position="bottom", ending_position="right"):
        """
        E---------------------------D
        L=========K     M===========N } beam_width
        F---------G  I  B-----------C
                  |  |  |
                  |  |  |
                  |  |  |
                  |  |  |
                  O  H   A
                  <----->
                column_width
        :param starting_position: can be one of 3 positions:
                1. bottom: H
                2. left: L
                3. right: N
        :param ending_position: can be one of 3 positions:
                1. bottom: H
                2. left: L
                3. right: N
                Difference lies in the bearing the robot is facing
        :return: static path points to follow np.array
        """
        # There are three basic lengths for which a combination of two define a static path
        # they are:
        #   1. HI = bottom
        #   2. LK = left
        #   3. MN = right
        allowed_position = ["left", "right", "bottom"]
        if not isinstance(starting_position, str):
            raise TypeError("starting_position must be <type str>")
        if not isinstance(ending_position, str):
            raise TypeError("starting_position must be <type, str>")
        if starting_position == ending_position:
            raise ValueError("starting_position cannot equal the ending_position")
        if starting_position not in allowed_position:
            raise ValueError("starting_position can only be 'left', 'right', 'bottom'")
        if ending_position not in allowed_position:
            raise ValueError("ending_position can only be 'left', 'right', 'bottom'")
        p0_noise = self.start_noise_scale * np.random.randn(2)
        starting_vector = {
            "bottom": [
                np.array([self._o[0] + (self.column_width / 2.0) + p0_noise[0], self._o[1] + p0_noise[1]]),
                np.array([self._o[0] + (self.column_width / 2.0), self._b[1]])
            ],
            "left": [
                np.array([self._f[0] + p0_noise[0], self._f[1] + (self.beam_width / 2) + p0_noise[1]]),
                np.array([self._g[0], self._g[1] + (self.beam_width / 2)]),
            ],
            "right": [
                np.array([self._c[0] + p0_noise[0], self._c[1] + (self.beam_width / 2) + p0_noise[1]]),
                np.array([self._b[0], self._b[1] + (self.beam_width / 2)]),
            ]
        }

        ending_vector = {
            "bottom": [
                np.array([self._o[0] + (self.column_width / 2.0), self._b[1]]),
                np.array([self._o[0] + (self.column_width / 2.0), self._o[1]])
            ],
            "left": [
                np.array([self._g[0], self._g[1] + (self.beam_width / 2)]),
                np.array([self._f[0], self._f[1] + (self.beam_width / 2)])
            ],
            "right": [
                np.array([self._b[0], self._b[1] + (self.beam_width / 2)]),
                np.array([self._c[0], self._c[1] + (self.beam_width / 2)]),
            ]
        }
        start_vector = starting_vector[starting_position]
        end_vector = ending_vector[ending_position]

        start_bearing = Bearing.starting[starting_position]
        end_bearing = Bearing.ending[ending_position]

        starting_vector_way_points = self._determine_intermediate_points_for_static(start_vector, start_bearing)
        ending_vector_way_points = self._determine_intermediate_points_for_static(end_vector, end_bearing)

        return np.array(starting_vector_way_points + ending_vector_way_points)

    @staticmethod
    def _determine_intermediate_points_for_static(vector_pnts, bearing):
        """
        Using parametric equation of a straight line, determine more points between a start and end point
        :param vector_pnts: [np.array([x_1, y_1]), np.array([x_2, y_2])]
        :param bearing: float value in radians
        :return: way_points, list of intermediate points
        """
        num_points = 150
        parametric_t = np.linspace(0, 1, num=num_points)
        start_pnt = vector_pnts[0]
        end_pnt = vector_pnts[1]

        parallel_vector = end_pnt - start_pnt

        x = start_pnt[0] + parametric_t * parallel_vector[0]
        y = start_pnt[1] + parametric_t * parallel_vector[1]

        way_points = []
        for idx in range(num_points):
            val = np.array([x[idx], y[idx], bearing])
            way_points.append(val)
        return way_points
