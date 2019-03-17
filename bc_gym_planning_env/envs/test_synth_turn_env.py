from bc_gym_planning_env.envs.synth_turn_env import ColoredEgoCostmapRandomAisleTurnEnv,\
    ColoredCostmapRandomAisleTurnEnv, RandomAisleTurnEnv
import pickle
import numpy as np


def test_colored_ego_costmap_random_aisle_turn_env():
    with open('ground_truth_1.pkl', 'rb') as f:
        ground_truth = pickle.load(f)
    env = ColoredEgoCostmapRandomAisleTurnEnv()
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
    with open('ground_truth_2.pkl', 'rb') as f:
        ground_truth = pickle.load(f)
    env = ColoredCostmapRandomAisleTurnEnv()
    env.reset()
    for step_snapshot in ground_truth:
        action = step_snapshot['action']
        observation, reward, done, _ = env.step(action)

        np.testing.assert_array_almost_equal(observation, step_snapshot['observation'])
        np.testing.assert_array_almost_equal(reward, step_snapshot['reward'])
        np.testing.assert_array_almost_equal(done, step_snapshot['done'])

        if done:
            env.reset()


def test_random_aisle_turn_env():
    with open('ground_truth_3.pkl', 'rb') as f:
        ground_truth = pickle.load(f)
    env = RandomAisleTurnEnv()
    env.reset()
    for step_snapshot in ground_truth:
        action = step_snapshot['action']
        observation, reward, done, _ = env.step(action)

        np.testing.assert_array_almost_equal(observation, step_snapshot['observation'])
        np.testing.assert_array_almost_equal(reward, step_snapshot['reward'])
        np.testing.assert_array_almost_equal(done, step_snapshot['done'])


def record_new_ground_truth_1():
    ground_truth = []
    env = ColoredEgoCostmapRandomAisleTurnEnv()
    env.reset()
    action_sequence = list(range(11))
    for action in action_sequence:
        observation, reward, done, _ = env.step([action])
        ground_truth.append({'action': action,
                             'observation': observation,
                             'reward': reward,
                             'done': done})
    with open('ground_truth_1.pkl', 'wb') as f:
        pickle.dump(ground_truth, f, pickle.HIGHEST_PROTOCOL)


def record_new_ground_truth_2():
    ground_truth = []
    env = ColoredCostmapRandomAisleTurnEnv()
    env.reset()
    action_sequence = list(range(11))
    for action in action_sequence:
        observation, reward, done, _ = env.step([action])
        ground_truth.append({'action': action,
                             'observation': observation,
                             'reward': reward,
                             'done': done})
    with open('ground_truth_2.pkl', 'wb') as f:
        pickle.dump(ground_truth, f, pickle.HIGHEST_PROTOCOL)


def record_new_ground_truth_3():
    ground_truth = []
    env = RandomAisleTurnEnv()
    env.reset()
    action_sequence = list(range(11))
    for action in action_sequence:
        observation, reward, done, _ = env.step([action])
        ground_truth.append({'action': action,
                             'observation': observation,
                             'reward': reward,
                             'done': done})
    with open('ground_truth_3.pkl', 'wb') as f:
        pickle.dump(ground_truth, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    test_colored_ego_costmap_random_aisle_turn_env()
    test_colored_costmap_random_aisle_turn_env()
    # test_random_aisle_turn_env()

    # record_new_ground_truth_1()
    # record_new_ground_truth_2()
    # record_new_ground_truth_3()

