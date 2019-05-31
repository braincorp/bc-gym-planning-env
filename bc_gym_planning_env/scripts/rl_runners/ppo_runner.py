import pandas as pd
import cv2
import torch
import torch.optim as optim
import numpy as np
import pickle

from vel.rl.metrics import EpisodeRewardMetric
from vel.storage.streaming.stdout import StdoutStreaming
from vel.util.random import set_seed
from vel.modules.input.image_to_tensor import ImageToTensorFactory
from vel.rl.models.stochastic_policy_model import StochasticPolicyModelFactory, StochasticPolicyModel
from vel.rl.models.backbone.nature_cnn_two_tower import NatureCnnTwoTowerFactory
from vel.rl.models.deterministic_policy_model import DeterministicPolicyModel
from vel.rl.reinforcers.on_policy_iteration_reinforcer import OnPolicyIterationReinforcer, OnPolicyIterationReinforcerSettings
from vel.schedules.linear import LinearSchedule
from vel.rl.algo.policy_gradient.ppo import PpoPolicyGradient
from vel.rl.env_roller.step_env_roller import StepEnvRoller
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
    model = StochasticPolicyModelFactory(
        input_block=ImageToTensorFactory(),
        backbone=NatureCnnTwoTowerFactory(input_width=133, input_height=133, input_channels=1)
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
            batch_size=256,
            experience_replay=4,
            number_of_steps=128
        ),
        model=model,
        algo=PpoPolicyGradient(
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            max_grad_norm=0.01,
            discount_factor=0.99,
            gae_lambda=0.95,
            cliprange=cliprange
        ),
        env_roller=StepEnvRoller(
            environment=vec_env,
            device=device
        )
    )

    # Model optimizer
    optimizer = optim.Adam(reinforcer.model.parameters(), lr=1e-6, eps=1.0e-5)

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
    eval_results = []
    for i in range(1, num_epochs+1):
        epoch_info = EpochInfo(
            training_info=training_info,
            global_epoch_idx=i,
            batches_per_epoch=10,
            optimizer=optimizer
        )

        reinforcer.train_epoch(epoch_info)

        eval_result = evaluate_model(model, vec_env, device, takes=1)
        eval_results.append(eval_result)

        if i % 100 == 0:
            torch.save(model.state_dict(), 'tmp_checkout.data')
            with open('tmp_eval_results.pkl', 'wb') as f:
                pickle.dump(eval_results, f, 0)

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
    video_recorder = VideoRecorder()

    for i in range(takes):
        frames = []
        result = record_take(model, env, device)
        rewards.append(result['r'])
        lengths.append(result['l'])
        frames.append(result['frames'])

        if debug:
            video_recorder.save_as_video(frames)

    video_recorder.release()
    print(pd.DataFrame({'lengths': lengths, 'rewards': rewards}).describe())
    model.train(mode=True)
    return {'rewards': rewards, 'lengths': lengths}


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
        observation_tensor = _bc_observations_to_tensor(observation, device)
        if isinstance(model, StochasticPolicyModel):
            actions = model.step(observation_tensor, argmax_sampling=False)['actions'].cpu().numpy()
        elif isinstance(model, DeterministicPolicyModel):
            actions = model.step(observation_tensor)['actions'].cpu().numpy()
        else:
            raise NotImplementedError
        action_classes = []
        for i in range(actions.shape[0]):
            action_class = Action(command=actions[i])
            action_classes.append(action_class)
        observation, reward, done, epinfo = env_instance.step(action_classes)
        steps += 1
        rewards += reward
        if debug or device.type == 'cpu':
            frames.append(env_instance.render(mode='human'))

        if done:
            print("episode reward: {}, steps: {}, collide: {}".format(rewards, steps, epinfo[0]['episode']['collide']))
            return {'r': rewards, 'l': steps, 'frames': frames}


def _bc_observations_to_tensor(observations, device):
    """Convert numpy array to a tensor
    :param observations dict: a dictionary of np.array
    :param device: put tensor on cpu or gpu
    :return: stack all observations into one torch tensors
    """
    if isinstance(observations, dict):
        input1 = observations['environment']
        input2 = observations['goal']
        input_additional_channel = np.zeros(input1.shape[:-1]+(1,))
        input_additional_channel[:, 0, 0:5, :] = input2
        input = np.concatenate((input1.astype(float), input_additional_channel), axis=3)
        return torch.from_numpy(input).to(device)
    else:
        raise NotImplementedError


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

    model = StochasticPolicyModelFactory(
        input_block=ImageToTensorFactory(),
        backbone=NatureCnnTwoTowerFactory(input_width=133, input_height=133, input_channels=1)
    ).instantiate(action_space=vec_env.action_space)
    model_checkpoint = torch.load('tmp_checkout.data', map_location='cpu')
    model.load_state_dict(model_checkpoint)

    evaluate_model(model, vec_env, device, takes=10)


class VideoRecorder(object):
    def __init__(self):
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._video_shape = (400, 600)
        self._out = cv2.VideoWriter('output.avi', fourcc, 100.0, self._video_shape)

    def save_as_video(self, frames):
        """function to record a demo video
        :param frames list[np.array]:  a list of images
        :return: None, video saved as a file
        """
        for trial in frames:
            for frame in trial:
                # frame = frame[0]
                frame = cv2.resize(frame, self._video_shape)
                # write the flipped frame
                self._out.write(frame)

                # cv2.imshow('frame', frame)
                # cv2.waitKey(1)

    def release(self):
        # Release everything if job is finished
        self._out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    train_model()
    eval_model()
