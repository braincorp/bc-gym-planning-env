from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import pickle
import numpy as np
import requests

from bc_gym_planning_env.utilities.artifacts_utils import get_cache_key_path
from bc_gym_planning_env.envs.synth_turn_env import ColoredEgoCostmapRandomAisleTurnEnv,\
    ColoredCostmapRandomAisleTurnEnv


def test_colored_ego_costmap_random_aisle_turn_env():
    session_tar_key = 'test_colored_ego_costmap_random_aisle_turn_env_ground_truth.pkl'
    session_tar_file = get_cache_key_path(session_tar_key)
    r = requests.get(
        'https://s3-us-west-1.amazonaws.com/braincorp-research-public-datasets/' + session_tar_key
    )
    with open(session_tar_file, 'wb') as f:
        f.write(r.content)

    with open(session_tar_file, 'rb') as f:
        ground_truth = pickle.load(f)
    env = ColoredEgoCostmapRandomAisleTurnEnv()
    env.seed(1001)
    env.reset()
    for step_snapshot in ground_truth:
        action = step_snapshot['action']
        observation, reward, done, _ = env.step(action)

        np.testing.assert_array_almost_equal(observation['environment'], step_snapshot['observation']['environment'])
        np.testing.assert_array_almost_equal(observation['goal'], step_snapshot['observation']['goal'])
        np.testing.assert_array_almost_equal(reward, step_snapshot['reward'])
        np.testing.assert_array_almost_equal(done, step_snapshot['done'])

        if done:
            env.reset()


def test_colored_costmap_random_aisle_turn_env():
    session_tar_key = 'test_colored_costmap_random_aisle_turn_env_ground_truth.pkl'
    session_tar_file = get_cache_key_path(session_tar_key)
    r = requests.get(
        'https://s3-us-west-1.amazonaws.com/braincorp-research-public-datasets/' + session_tar_key
    )
    with open(session_tar_file, 'wb') as f:
        f.write(r.content)

    with open(session_tar_file, 'rb') as f:
        ground_truth = pickle.load(f)
    env = ColoredCostmapRandomAisleTurnEnv()
    env.seed(1001)
    env.reset()
    for step_snapshot in ground_truth:
        action = step_snapshot['action']
        observation, reward, done, _ = env.step(action)

        np.testing.assert_array_almost_equal(observation, step_snapshot['observation'])
        np.testing.assert_array_almost_equal(reward, step_snapshot['reward'])
        np.testing.assert_array_almost_equal(done, step_snapshot['done'])

        if done:
            env.reset()


def record_new_ground_truth_for_colored_ego_costmap_random_aisle_turn_env():
    ground_truth = []
    env = ColoredEgoCostmapRandomAisleTurnEnv()
    env.seed(1001)
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        ground_truth.append({'action': action,
                             'observation': observation,
                             'reward': reward,
                             'done': done})
        if done:
            env.reset()

    session_tar_key = 'test_colored_ego_costmap_random_aisle_turn_env_ground_truth.pkl'
    with open(session_tar_key, 'wb') as f:
        pickle.dump(ground_truth, f, protocol=0)
    # to update ground truth, need to manually upload this file to AWS


def record_new_ground_truth_for_colored_costmap_random_aisle_turn_env():
    ground_truth = []
    env = ColoredCostmapRandomAisleTurnEnv()
    env.seed(1001)
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        ground_truth.append({'action': action,
                             'observation': observation,
                             'reward': reward,
                             'done': done})
        if done:
            env.reset()

    session_tar_key = 'test_colored_costmap_random_aisle_turn_env_ground_truth.pkl'
    with open(session_tar_key, 'wb') as f:
        pickle.dump(ground_truth, f, protocol=0)
    # to update ground truth, need to manually upload this file to AWS


if __name__ == '__main__':
    test_colored_ego_costmap_random_aisle_turn_env()
    test_colored_costmap_random_aisle_turn_env()
    # record_new_ground_truth_for_colored_ego_costmap_random_aisle_turn_env()
    # record_new_ground_truth_for_colored_costmap_random_aisle_turn_env()
