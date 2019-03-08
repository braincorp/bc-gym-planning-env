from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

from bc_gym_planning_env.utilities.coordinate_transformations import world_to_pixel
from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.utilities.map_drawing_utils import get_drawing_coordinates_from_physical, \
    get_drawing_angle_from_physical, draw_world_map, draw_wide_path, prepare_canvas
from bc_gym_planning_env.utilities.path_tools import get_pixel_footprint, inscribed_radius


class PlanningEnvironment(object):
    """
    A simulation environment in which to run a robot.  Control signals are passed in via
    the step function, and this object updates the state of the world.
    """

    def __init__(self, costmap, robot, robot_footprint_scaling=1, on_collision_callback=lambda *args: None):

        """
        :param costmap: A CostMap2D defining the obstacles in the environment.,
        :param robot: An IRobot object
        :param robot_footprint_scaling: Added to pass test_force_small_clearance after new accurate distance transform
        :param on_collision_callback: A function of the form
            on_collision_callback(position)
            That is called when a collision occurs.
        """
        assert isinstance(costmap, CostMap2D), ("Please initialize this class with Costmap2d instance\n"
                                                "e.g. CostMap2D.create_from_yaml(artifact_path('empty_plane_groundtruth.yaml'))")
        self._costmap = costmap
        self._robot = robot
        self._robot_footprint_scaling = robot_footprint_scaling
        self._new_robot_position = None
        self._new_robot_goal = None
        self._on_collision_callback = on_collision_callback

    def draw_robot(self, image, x, y, angle, color, costmap):
        px, py = get_drawing_coordinates_from_physical(costmap.get_data().shape,
                                                       costmap.get_resolution(),
                                                       costmap.get_origin(),
                                                       (x, y))
        pangle = get_drawing_angle_from_physical(angle)
        self._robot.draw(image, px, py, pangle, color, self._costmap.get_resolution())

    def step(self, dt, control_signals, check_collisions=True):
        if self._new_robot_position:
            try:
                self.set_robot_pose(*self._new_robot_position)
            except ValueError:
                pass
            self._new_robot_position = None
        old_position = self._robot.get_pose()
        self._robot.step(dt, control_signals)
        new_position = self._robot.get_pose()
        if check_collisions:
            collides = self._pose_collides(*new_position)
            if collides:
                self._on_collision_callback(new_position)
                self._robot.set_pose(*old_position)

    def draw(self, path_to_follow, original_path):
        """
        Draw obstacles, path and a robot
        :param path_to_follow: numpy array of (x, y, angle) of a path to follow
        :return: numpy BGR rendered image
        """
        img = prepare_canvas(self._costmap.get_data().shape)

        draw_wide_path(
            img, original_path,
            robot_width=2*inscribed_radius(self._robot.get_footprint()),
            origin=self._costmap.get_origin(),
            resolution=self._costmap.get_resolution(),
            color=(240, 240, 240)
        )
        draw_wide_path(
            img, path_to_follow,
            robot_width=2*inscribed_radius(self._robot.get_footprint()),
            origin=self._costmap.get_origin(),
            resolution=self._costmap.get_resolution(),
            color=(220, 220, 220)
        )
        draw_world_map(img, self._costmap.get_data())

        x, y, angle = self._robot.get_pose()
        self.draw_robot(img, x, y, angle, color=(0, 100, 0), costmap=self._costmap)
        return img

    def _pose_collides(self, x, y, angle):
        '''
        Check if robot footprint at x, y (world coordinates) and
            oriented as yaw collides with lethal obstacles.
        '''
        kernel_image = get_pixel_footprint(angle,
                                           self._robot.get_footprint() * self._robot_footprint_scaling,
                                           self._costmap.get_resolution())
        # Get the coordinates of where the footprint is inside the kernel_image (on pixel coordinates)
        kernel = np.where(kernel_image)
        # Move footprint to (x,y), all in pixel coordinates
        x, y = world_to_pixel(np.array([x, y]), self._costmap.get_origin(), self._costmap.get_resolution())
        collisions = y + kernel[0] - kernel_image.shape[0] // 2, x + kernel[1] - kernel_image.shape[1] // 2
        raw_map = self._costmap.get_data()
        # Check if the footprint pixel coordinates are valid, this is, if they are not negative and are inside the map
        good = np.logical_and(np.logical_and(collisions[0] >= 0, collisions[0] < raw_map.shape[0]),
                              np.logical_and(collisions[1] >= 0, collisions[1] < raw_map.shape[1]))

        # Just from the footprint coordinates that are good, check if they collide
        # with obstacles inside the map
        return bool(np.any(raw_map[collisions[0][good],
                                   collisions[1][good]] == CostMap2D.LETHAL_OBSTACLE))

    def set_robot_pose(self, x, y, angle, check=True):
        if check:
            if self._pose_collides(x, y, angle):
                raise ValueError("This pose collides with obstacles")
        self._robot.set_pose(x, y, angle)

    def get_robot_pose(self):
        return self._robot.get_pose()

    def get_robot_state(self):
        return self._robot.get_robot_state()

    def get_robot(self):
        return self._robot
