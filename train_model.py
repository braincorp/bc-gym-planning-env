import pandas as pd
import cv2
import torch
import torch.optim as optim
import numpy as np

from vel.rl.metrics import EpisodeRewardMetric
from vel.storage.streaming.stdout import StdoutStreaming
from vel.util.random import set_seed
from vel.rl.models.policy_gradient_model import PolicyGradientModelFactory, PolicyGradientModel
from vel.rl.models.backbone.nature_cnn_two_tower import NatureCnnTwoTowerFactory
from vel.rl.models.deterministic_policy_model import DeterministicPolicyModel
from vel.rl.reinforcers.on_policy_iteration_reinforcer import OnPolicyIterationReinforcer, OnPolicyIterationReinforcerSettings
from vel.schedules.linear import LinearSchedule
from vel.rl.algo.policy_gradient.ppo import PpoPolicyGradient
from vel.rl.env_roller.vec.step_env_roller import StepEnvRoller
from vel.api.info import TrainingInfo, EpochInfo
from vel.rl.commands.rl_train_command import FrameTracker
from vel.openai.baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from bc_gym_planning_env.envs.synth_turn_env import ColoredEgoCostmapRandomAisleTurnEnv
from bc_gym_planning_env.envs.base.action import Action


def train_model():
    """a sample training script, that creates a PPO instance and train it with bc-gym environment
    :return: None
    """
    device = torch.device('cpu')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)
    env_function = lambda: ColoredEgoCostmapRandomAisleTurnEnv()
    vec_env = DummyVecEnv([env_function])

    # Again, use a helper to create a model
    # But because model is owned by the reinforcer, model should not be accessed using this variable
    # but from reinforcer.model property
    model = PolicyGradientModelFactory(
        backbone=NatureCnnTwoTowerFactory(input_width=133, input_height=117, input_channels=1)
    ).instantiate(action_space=vec_env.action_space)

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
    """evaluate the performance of a rl model with a given environment
    :param model: a trained rl model
    :param env: environment
    :param device: cpu or gpu
    :param takes: number of trials/rollout
    :param debug: record a video in debug mode
    :return: None
    """
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
    """run one rollout of the rl model with the environment, until done is true
    :param model: rl policy model
    :param env_instance: an instance of the environment to be evaluated
    :param device: cpu or gpu
    :param debug: debug mode has gui output
    :return: some basic metric info of this rollout
    """
    frames = []
    steps = 0
    rewards = 0
    observation = env_instance.reset()

    print("Evaluating environment...")

    while True:
        observation_tensor = _dict_to_tensor(observation, device)
        if isinstance(model, PolicyGradientModel):
            actions = model.step(observation_tensor, argmax_sampling=False)['actions'].to(device)[0]
        elif isinstance(model, DeterministicPolicyModel):
            actions = model.step(observation_tensor)['actions'].to(device)[0]
        else:
            raise NotImplementedError
        action_class = Action(command=actions.cpu().numpy())
        observation, reward, done, epinfo = env_instance.step(action_class)
        steps += 1
        rewards += reward
        if debug or device.type == 'cpu':
            frames.append(env_instance.render(mode='human'))

        if done:
            print("episode reward: {}, steps: {}".format(rewards, steps))
            return {'r': rewards, 'l': steps, 'frames': frames}


def _dict_to_tensor(numpy_array_dict, device):
    """Convert numpy array to a tensor
    :param numpy_array_dict dict: a dictionary of np.array
    :param device: put tensor on cpu or gpu
    :return: a dictionary of torch tensors
    """
    if isinstance(numpy_array_dict, dict):
        torch_dict = {}
        for k, v in numpy_array_dict.items():
            torch_dict[k] = torch.from_numpy(numpy_array_dict[k]).to(device)
        return torch_dict
    else:
        return torch.from_numpy(numpy_array_dict).to(device)


def eval_model():
    """load a checkpoint data and evaluate its performance
    :return: None
    """
    device = torch.device('cpu')
    seed = 1001

    # Set random seed in python std lib, numpy and pytorch
    set_seed(seed)

    env_function = lambda: ColoredEgoCostmapRandomAisleTurnEnv()
    vec_env = DummyVecEnv([env_function])
    vec_env.reset()

    model = PolicyGradientModelFactory(
        backbone=NatureCnnTwoTowerFactory(input_width=133, input_height=117, input_channels=1)
    ).instantiate(action_space=vec_env.action_space)
    model_checkpoint = torch.load('tmp_checkout.data', map_location='cpu')
    model.load_state_dict(model_checkpoint)

    evaluate_model(model, vec_env, device, takes=10)


def save_as_video(frames):
    """function to record a demo video
    :param frames list[np.array]:  a list of images
    :return: None, video saved as a file
    """
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
