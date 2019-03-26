"""Functions to easy create standard reward provider state"""
from __future__ import print_function
from __future__ import absolute_import

from bc_gym_planning_env.envs.base.reward_provider_examples import RewardProviderStateExamples, RewardProviderExamples
from bc_gym_planning_env.envs.base.reward import \
    ContinuousRewardPurePursuitProviderState, ContinuousRewardProviderState,\
    ContinuousRewardPurePursuitProvider, ContinuousRewardProvider


def create_reward_provider_state(reward_provider_state_name, **kwargs):
    """
    Given a robot name (along with construction parameters common to all robots), create a new robot.

    :param reward_provider_state_name: A string reward provider state name: One of the RewardProviderStateExamples enum
    :param kwargs: misc arguments
    :return: A reward provider state object
    """
    if reward_provider_state_name == RewardProviderStateExamples.CONTINUOUS_REWARD_STATE:
        robot = ContinuousRewardProviderState(**kwargs)
        return robot
    elif reward_provider_state_name == RewardProviderStateExamples.CONTINUOUS_REWARD_PURE_PURSUIT_STATE:
        return ContinuousRewardPurePursuitProviderState(**kwargs)
    else:
        raise Exception('No reward provider state name "{}" exists'.format(reward_provider_state_name))


def get_reward_provider_example(reward_provider_name):
    """
    Get class corresponding to reward_provider_name

    :param reward_provider_name: reward provider name string (see below for valid inputs)
    :return reward provider instance: an instance of a particular type of reward
    """
    name_to_reward_provider = \
        {RewardProviderExamples.CONTINUOUS_REWARD: ContinuousRewardProvider,
         RewardProviderExamples.CONTINUOUS_REWARD_PURE_PURSUIT: ContinuousRewardPurePursuitProvider}

    valid_reward_provider_types = list(name_to_reward_provider.keys())

    if reward_provider_name not in valid_reward_provider_types:
        raise AssertionError("Unknown reward provider: {}. Should be one of {}".format(reward_provider_name,
                                                                                       valid_reward_provider_types))

    return name_to_reward_provider[reward_provider_name]
