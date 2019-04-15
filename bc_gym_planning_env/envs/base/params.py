""" Classes and utils for support of Brain Corp planing env parametrization"""
from __future__ import absolute_import
from __future__ import division

import attr
import numpy as np

from bc_gym_planning_env.robot_models.standard_robot_names_examples import StandardRobotExamples
from bc_gym_planning_env.utilities.serialize import Serializable
from bc_gym_planning_env.envs.base.reward import RewardParams
from bc_gym_planning_env.envs.base.reward_provider_examples import RewardProviderExamples


@attr.s(frozen=True)
class EnvParams(Serializable):
    """ Parametrization of the environment.  """
    dt = attr.ib(type=float, default=0.05)                     # how much time passes between two observations
    goal_ang_dist = attr.ib(type=float, default=np.pi/2)       # how close angularly to goal to reach it
    goal_spat_dist = attr.ib(type=float, default=1.0)          # how close to goal to reach it
    initial_wheel_angle = attr.ib(default=0.0, type=float)     # initialization of wheel angle
    iteration_timeout = attr.ib(type=int, default=1200)        # how many timesteps to reach the goal
    path_limiter_max_dist = attr.ib(type=float, default=5.0)   # spatial horizon of path follower
    robot_name = attr.ib(
        default=StandardRobotExamples.INDUSTRIAL_TRICYCLE_V1)             # name of the robot, e.g. determines footprint
    resolution = attr.ib(default=0.03, type=float)             # spatial resolution
    refine_path = attr.ib(default=True, type=bool)            # should we make path dense?
    path_delta = attr.ib(default=0.05, type=float)             # if we want to make path dense, make a point every 5cm
    pose_delay = attr.ib(default=0, type=int)                  # we perceive poses with how much delay
    control_delay = attr.ib(default=0, type=int)               # how much delay in perceived controls
    state_delay = attr.ib(default=0, type=int)                 # state perception delay, in reality ~ 0.11s (2 steps)

    reward_provider_name = attr.ib(                            # name of the reward provider
        default=RewardProviderExamples.CONTINUOUS_REWARD_PURE_PURSUIT)
    reward_provider_params = attr.ib(                          # parameters of the reward provider
        default=attr.Factory(
            lambda self: RewardParams(
                spatial_precision=self.goal_spat_dist,
                angular_precision=self.goal_ang_dist
            ),
            takes_self=True
        )
    )

    VERSION = 1

    def serialize(self):
        # pylint: disable=no-member
        resu = attr.asdict(self)
        resu['version'] = self.VERSION
        resu['reward_provider_params'] = self.reward_provider_params.serialize()
        return resu

    @classmethod
    def deserialize(cls, state):
        ver = state.pop('version')
        assert ver == cls.VERSION
        state['reward_provider_params'] = RewardParams.deserialize(state['reward_provider_params'])
        return cls(**state)
