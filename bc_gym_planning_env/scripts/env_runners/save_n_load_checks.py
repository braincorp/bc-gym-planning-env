""" A code to test if serialization, deserialization and equality testing are working well for
 elements of the planning environment stack. """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import pickle
import time

from bc_gym_planning_env.envs.base.env import PlanEnv
from bc_gym_planning_env.envs.base.obs import Observation
from bc_gym_planning_env.envs.base.params import EnvParams
from bc_gym_planning_env.envs.base.maps import example_config, generate_trajectory_and_map_from_config


def identity_serialize_deserialize(thing, the_class):
    """ Take the thing, serialize it to a dict of basic types, pickle it.
    Then unpickle the result and deserialize it to the original object.
    Return the result. This whole operation should be equivalent to identity.
    It "shouldn't change" the thing.

    :param thing Serializable: the thing to test
    :param the_class type: final type of serializable
    :return the_class: the deserialized object
    """
    with open('env.pkl', 'wb') as f:
        data = thing.serialize()
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('env.pkl', 'rb') as f:
        serialized = pickle.load(f)

    return the_class.deserialize(serialized)


if __name__ == '__main__':
    path, costmap = generate_trajectory_and_map_from_config(example_config())

    env = PlanEnv(
        costmap=costmap,
        path=path,
        params=EnvParams()
    )

    env.reset()

    t = time.time()
    env_two = identity_serialize_deserialize(env, PlanEnv)

    actions = [env.action_space.sample() for _ in range(100)]

    for action in actions:
        pre_step_state_1 = env.get_state()
        pre_step_state_2 = env_two.get_state()

        assert pre_step_state_1 == pre_step_state_2

        obs1, r1, done1, _ = env.step(action)
        obs2, r2, done2, _ = env_two.step(action)

        obs3 = identity_serialize_deserialize(obs2, Observation)

        post_step_state_1 = env.get_state()
        post_step_state_2 = env_two.get_state()

        assert post_step_state_1 == post_step_state_2

        assert obs1 == obs2
        assert obs1 == obs3
        assert r1 == r2
        assert done1 == done2
