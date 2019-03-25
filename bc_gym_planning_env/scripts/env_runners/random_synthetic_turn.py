"""
Simple script for running random turn environment.
Do some random actions, but biased going forward.
 """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


from bc_gym_planning_env.envs.synth_turn_env import RandomAisleTurnEnv
from bc_gym_planning_env.envs.egocentric import EgocentricCostmap


if __name__ == '__main__':
    env = RandomAisleTurnEnv()
    env = EgocentricCostmap(env)

    initial_state = env.get_state()

    import time
    while True:
        env.set_state(initial_state)
        env.render()

        done = False

        while not done:
            command = env.action_space.sample()
            command.command[0] += 0.3
            # print("Applying {}".format(command))
            obs, _, done, _ = env.step(command)
            env.render()
            time.sleep(0.05)
