""" Module that handles some static map constructing logic helper. """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import attr
import cv2
import numpy as np


from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.utilities.map_drawing_utils import add_wall_to_static_map
from bc_gym_planning_env.utilities.path_tools import refine_path, orient_path


@attr.s
class MapConfig(object):
    """ Configuration that can be turned into
    (path to follow, costmap) pair """
    trajectory = attr.ib()
    obstacles = attr.ib(type=list)
    size = attr.ib(type=tuple)
    origin = attr.ib(type=tuple)
    resolution = attr.ib(type=float)


@attr.s
class Wall(object):
    """ The most basic type of obstacle - a wall between two points. """
    from_pt = attr.ib(type=np.ndarray)
    to_pt = attr.ib(type=np.ndarray)

    def render(self, costmap):
        """
        Render this wall onto a costmap.
        :param costmap Costmap2D: Costmap to draw the wall on.
        :return Costmap2D: the costmap with the wall on it
        """
        add_wall_to_static_map(costmap, self.from_pt, self.to_pt, width=costmap.get_resolution()*4)
        return costmap


def generate_zigzag_walls(trajectory, corridor_y_span, corridor_x_span):
    """Generates list of objects corresponding to zig-zagging walls.
    :param trajectory np.ndarray(N, 3): a zig-zagging trajectory around which
                                        the parallel walls should be drawn
    :param corridor_y_span float: how high should the corridor be on the costmap
    :param corridor_x_span float: how wide should the corridor be?
    :return List[Wall]: list of obstacles to draw on the costmap
    """
    obstacles = []

    stuff = trajectory + np.array([-corridor_y_span, corridor_x_span])
    for from_pt, to_pt in zip(stuff[:-1], stuff[1:]):
        wall = Wall(from_pt=from_pt, to_pt=to_pt)
        obstacles.append(wall)

    stuff = trajectory + np.array([corridor_y_span, -corridor_x_span])
    for from_pt, to_pt in zip(stuff[:-1], stuff[1:]):
        wall = Wall(from_pt=from_pt, to_pt=to_pt)
        obstacles.append(wall)

    return obstacles


def generate_trajectory_and_map_from_config(config):
    """ Based on given MapConfig,
    generate (trajectory to follow, CostMap2D) pair.
    :param config MapConfig: map config based on which we will make a map.
    :return (path_to_follow, CostMap2D) pair: path following task specification
    """

    static_map = CostMap2D.create_empty(
        world_size=config.size,
        resolution=config.resolution,
        world_origin=config.origin
    )

    for obs in config.obstacles:
        static_map = obs.render(static_map)

    return config.trajectory, static_map


def example_config():
    """
    Generates an example config.
    :return MapConfig: an example config
    """
    traj = np.array([
        [2., 2.],
        [2., 4.],
        [4., 4.],
        [6., 4.],
        [6., 8.],
    ])

    obs = generate_zigzag_walls(
        trajectory=traj,
        corridor_y_span=0.65,
        corridor_x_span=0.975
    )

    traj = refine_path(orient_path(traj), 0.05)

    config = MapConfig(
        trajectory=traj,
        obstacles=obs,
        size=(10, 10),
        resolution=0.03,
        origin=(0, 0)
    )

    return config


def _neighbours():
    """
    What to add to my [x,y] coordinates to get
    coordinates of all my neighbours?
    :return List[List[int]]: What to add to my [x,y] coordinates to get
    coordinates of all my neighbours?
    """

    return [
        [0, 1],
        [1, 1],
        [1, 0],
        [1, -1],
        [0, -1],
        [-1, -1],
        [-1, 0],
        [-1, 1],
    ]


def load_costmap_from_img(img_fname):
    """
    Make a costmap from an image.
    :param img_fname str: fname of image to load.
    :return (path_to_follow, CostMap2D) pair: path following task specification
    """
    resolution = 0.03
    world_size = 10

    static_map = CostMap2D.create_empty(
        world_size=(world_size, world_size),
        resolution=resolution,
        world_origin=(0, 0)
    )

    img = cv2.imread(img_fname)
    # opencv loads bgr
    b, g, _ = img[..., 0], img[..., 1], img[..., 2]

    start_img = g
    # walls_img = r
    path_img = b

    start_y, start_x = np.where(start_img == 255)
    start_x = start_x[0]
    start_y = start_y[0]

    current_node = start_x, start_y
    path = []

    found_neighbour = True

    while found_neighbour:
        path.append(np.array(current_node))
        cur_x, cur_y = current_node

        found_neighbour = False
        for v_x, v_y in _neighbours():
            n_x = cur_x + v_x
            n_y = cur_y + v_y
            val = path_img[n_y, n_x]

            if val == 255:
                current_node = n_x, n_y
                found_neighbour = True
                path_img[n_y, n_x] = 0
                break

    prepath = np.array(path)
    path = prepath * resolution

    # Should actually do this, but not allowed for now
    # static_map._data = np.clip(walls_img, 0, 254)

    path = orient_path(path)
    return path, static_map
