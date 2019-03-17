import pandas as pd
import numpy as np
import cv2
import torch
import torch.optim as optim

from vel.rl.metrics import EpisodeRewardMetric
from vel.storage.streaming.stdout import StdoutStreaming
from vel.util.random import set_seed

from vel.rl.env.classic_atari import ClassicAtariEnv
from vel.rl.vecenv.subproc import SubprocVecEnvWrapper

from vel.rl.models.policy_gradient_model import PolicyGradientModelFactory, PolicyGradientModel
from vel.rl.models.backbone.nature_cnn_two_tower import NatureCnnTwoTowerFactory
from vel.rl.models.backbone.nature_cnn2 import NatureCnnFactory2

from vel.rl.models.deterministic_policy_model import DeterministicPolicyModelFactory, DeterministicPolicyModel
from vel.rl.models.backbone.mlp import MLPFactory

from vel.rl.reinforcers.on_policy_iteration_reinforcer import (
    OnPolicyIterationReinforcer, OnPolicyIterationReinforcerSettings
)
from vel.rl.reinforcers.buffered_single_off_policy_iteration_reinforcer import (
    BufferedSingleOffPolicyIterationReinforcer, BufferedSingleOffPolicyIterationReinforcerSettings
)

from vel.schedules.linear import LinearSchedule
from vel.rl.algo.policy_gradient.ppo import PpoPolicyGradient
from vel.rl.algo.policy_gradient.a2c import A2CPolicyGradient
from vel.rl.algo.policy_gradient.acer import AcerPolicyGradient
from vel.rl.algo.policy_gradient.ddpg import DeepDeterministicPolicyGradient

from vel.rl.env_roller.single.deque_replay_roller_ou_noise import DequeReplayRollerOuNoise

from vel.rl.env_roller.vec.step_env_roller import StepEnvRoller

from vel.api.info import TrainingInfo, EpochInfo
from vel.rl.commands.rl_train_command import FrameTracker

import vel.openai.baselines.common.cmd_util as butils

import gym
import bc_gym_planning_env
from bc_gym_planning_env.envs.synth_turn_env import RandomAisleTurnEnv
from bc_gym_planning_env.envs.synth_turn_env import ColoredEgoCostmapRandomAisleTurnEnv
from bc_gym_planning_env.envs.base.action import Action

from vel.openai.baselines import logger
from vel.openai.baselines.bench import Monitor


def train_model():
    device = torch.device('cpu')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)
    vec_env = butils.make_vec_env(
        env_id='EgoCostmapAsImgRandomTurnRoboPlanning-v0',
        env_type='robo_planning',
        num_env=1,
        seed=0,
        flatten_dict_observations=False
    )
    # vec_env.reset()
    # vec_env.step(vec_env.action_space.sample())

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model = PolicyGradientModelFactory(
        backbone=NatureCnnTwoTowerFactory(input_width=133, input_height=117, input_channels=1)
    ).instantiate(action_space=vec_env.action_space)

    # model_factory = DeterministicPolicyModelFactory(
    #     policy_backbone=NatureCnnFactory(input_width=133, input_height=117, input_channels=1),
    #     value_backbone=NatureCnnFactory2(input_width=133, input_height=117, input_channels=1),
    # )
    # model = model_factory.instantiate(action_space=vec_env.action_space)


    # Set schedule for gradient clipping.
    cliprange = LinearSchedule(
        initial_value=0.01,
        final_value=0.0
    )

    # Reinforcer - an object managing the learning process
    reinforcer = OnPolicyIterationReinforcer(
        device=device,
        settings=OnPolicyIterationReinforcerSettings(
            discount_factor=0.99,
            batch_size=256,
            experience_replay=4
        ),
        model=model,
        algo=PpoPolicyGradient(
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            max_grad_norm=0.02,
            cliprange=cliprange
        ),
        env_roller=StepEnvRoller(
            environment=vec_env,
            device=device,
            gae_lambda=0.95,
            number_of_steps=128,
            discount_factor=0.99,
        )
    )

    # Model optimizer
    optimizer = optim.Adam(reinforcer.model.parameters(), lr=5e-6, eps=1.0e-5)

    # Overall information store for training information
    training_info = TrainingInfo(
        metrics=[
            EpisodeRewardMetric('episode_rewards'),  # Calculate average reward from episode
        ],
        callbacks=[
            StdoutStreaming(),   # Print live metrics every epoch to standard output
            FrameTracker(1.1e8)      # We need frame tracker to track the progress of learning
        ]
    )

    # A bit of training initialization bookkeeping...
    training_info.initialize()
    reinforcer.initialize_training(training_info)
    training_info.on_train_begin()

    # Let's make 10 batches per epoch to average metrics nicely
    # Rollout size is 8 environments times 128 steps
    num_epochs = int(1.1e8 / (128 * 1) / 10)

    # Normal handrolled training loop
    for i in range(1, num_epochs+1):
        epoch_info = EpochInfo(
            training_info=training_info,
            global_epoch_idx=i,
            batches_per_epoch=10,
            optimizer=optimizer
        )

        reinforcer.train_epoch(epoch_info)
        if i % 1000 == 0:
            torch.save(model.state_dict(), 'tmp_checkout.data')
        evaluate_model(model, vec_env, device, takes=1)

    training_info.on_train_end()


def evaluate_model(model, env, device, takes=1, debug=False):
    model.eval()

    rewards = []
    lengths = []
    frames = []

    for i in range(takes):
        result = record_take(model, env, device)
        rewards.append(result['r'])
        lengths.append(result['l'])
        frames.append(result['frames'])

    if debug:
        save_as_video(frames)
    print(pd.DataFrame({'lengths': lengths, 'rewards': rewards}).describe())
    model.train(mode=True)


@torch.no_grad()
def record_take(model, env_instance, device, debug=False):
    frames = []
    steps = 0
    rewards = 0

    observation = env_instance.reset()

    # frames.append(env_instance.render('rgb_array'))

    print("Evaluating environment...")

    while True:
        # observation_array = np.expand_dims(np.array(observation), axis=0)
        # observation_tensor = torch.from_numpy(observation_array).to(device)
        observation_tensor = _dict_to_tensor(observation, device)
        if isinstance(model, PolicyGradientModel):
            actions = model.step(observation_tensor, argmax_sampling=False)['actions'].to(device)[0]
        elif isinstance(model, DeterministicPolicyModel):
            actions = model.step(observation_tensor)['actions'].to(device)[0]
        else:
            raise NotImplementedError
        # print("actions: {}, observation: {}".format(actions.cpu().numpy(), np.ravel(observation_tensor['goal'].cpu().numpy())))
        action_class = Action(command=actions.cpu().numpy())
        observation, reward, done, epinfo = env_instance.step(action_class)
        # print("reward: {}".format(reward))
        steps += 1
        rewards += reward
        if debug or device.type == 'cpu':
            frames.append(env_instance.render(mode='human'))

        if done:
            print("episode reward: {}, steps: {}".format(rewards, steps))
            return {'r': rewards, 'l': steps, 'frames': frames}


def _dict_to_tensor(numpy_array_dict, device):
    """ Convert numpy array to a tensor """
    if isinstance(numpy_array_dict, dict):
        torch_dict = {}
        for k, v in numpy_array_dict.items():
            torch_dict[k] = torch.from_numpy(numpy_array_dict[k]).to(device)
        return torch_dict
    else:
        return torch.from_numpy(numpy_array_dict).to(device)


def eval_model():
    device = torch.device('cpu')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    vec_env = butils.make_vec_env(
        env_id='EgoCostmapAsImgRandomTurnRoboPlanning-v0',
        env_type='robo_planning',
        num_env=1,
        seed=0,
        flatten_dict_observations=False
    )
    vec_env.reset()

    model = PolicyGradientModelFactory(
        backbone=NatureCnnTwoTowerFactory(input_width=133, input_height=117, input_channels=1)
    ).instantiate(action_space=vec_env.action_space)
    model_checkpoint = torch.load('tmp_checkout.data', map_location='cpu')
    model.load_state_dict(model_checkpoint)

    evaluate_model(model, vec_env, device, takes=10)


def save_as_video(frames):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_shape = (400, 600)
    out = cv2.VideoWriter('output.avi', fourcc, 400.0, video_shape)

    for trial in frames:
        for frame in trial:
            frame = frame[0]
            frame = cv2.resize(frame, video_shape)
            # write the flipped frame
            out.write(frame)

            cv2.imshow('frame', frame)
            cv2.waitKey(1)

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    train_model()
    eval_model()
