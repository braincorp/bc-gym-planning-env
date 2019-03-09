from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


from bc_gym_planning_env.envs.base.action import Action
from bc_gym_planning_env.envs.base.draw import draw_robot
from bc_gym_planning_env.envs.base.env import _pose_collides
from bc_gym_planning_env.utilities.costmap_2d import CostMap2D
from bc_gym_planning_env.utilities.map_drawing_utils import draw_world_map, draw_wide_path, prepare_canvas
from bc_gym_planning_env.utilities.path_tools import inscribed_radius


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

    def step(self, dt, control_signals, check_collisions=True):
        if self._new_robot_position:
            try:
                self.set_robot_pose(*self._new_robot_position)
            except ValueError:
                pass
            self._new_robot_position = None
        old_position = self._robot.get_pose()
        self._robot.step(dt, Action(command=control_signals))
        new_position = self._robot.get_pose()
        if check_collisions:
            collides = _pose_collides(new_position[0], new_position[1], new_position[2], self._robot, self._costmap)
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
        draw_robot(self._robot, img, x, y, angle, color=(0, 100, 0), costmap=self._costmap)
        return img

    def set_robot_pose(self, x, y, angle, check=True):
        if check:
            if _pose_collides(x, y, angle, self._robot, self._costmap):
                raise ValueError("This pose collides with obstacles")
        self._robot.set_pose(x, y, angle)

    def get_robot_pose(self):
        return self._robot.get_pose()

    def get_robot_state(self):
        return self._robot.get_robot_state()

    def get_robot(self):
        return self._robot
