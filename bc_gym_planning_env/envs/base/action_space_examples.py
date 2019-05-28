""" Names of different action spaces. """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from bc_gym_planning_env.envs.base import spaces


class ActionSpaceExamples:
    """ Names of action space"""
    CONTINUOUS_SPACE = 'continuous_space'
    DISCRETE_SPACE = 'discrete_space'


def get_action_space_example(action_space_name):
    """
    Get parameters corresponding to action_space_name

    :param action_space_name: action space name string (see below for valid inputs)
    :return reward action space instance: an instance of a particular type of reward
    """
    if action_space_name == ActionSpaceExamples.DISCRETE_SPACE:
        action_space = spaces.Discrete(11)
        action_command_list = {0: [0.2, 0.0],
                               1: [0.2, 0.3],
                               2: [0.2, -0.3],
                               3: [0.2, 0.5],
                               4: [0.2, -0.5],
                               5: [0.2, 0.7],
                               6: [0.2, -0.7],
                               7: [0.2, 1.1],
                               8: [0.2, -1.1],
                               9: [0.2, 1.3],
                               10: [0.2, -1.3]}
        return action_space, action_command_list
    elif action_space_name == ActionSpaceExamples.CONTINUOUS_SPACE:
        action_space = spaces.Box(
            low=np.array([0.1, -1.5]),
            high=np.array([0.5, 1.5]),
            dtype=np.float32)
        return action_space, None
    else:
        raise AssertionError("Unknown action space: {}.".format(action_space_name))

