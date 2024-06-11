from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dill
from wrappers import gym_wrapper

import gymnasium as gym
from env import FourRoomsEnv

class CNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64, load_file = None, freeze_linear = False):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.image_normaliser = 1
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
        )


        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = np.prod(self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1:])

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())


    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations / self.image_normaliser
        x = self.cnn(observations)
        x = x.flatten(start_dim=1)
        return self.linear(x)

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open('configs/fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)

num_train_configs = len(train_config['topologies'])

exp_frac = 1.0
buffer_size = 500_000
batch_size = 256
tau = 0.01
gamma = .99
max_grad_norm = 1
gradient_steps = 1
target_update_interval = 100
train_freq = 10
exploration_final_eps = 0.1
learning_rate = 0.0001
n_envs = 10
device = "cuda" if th.cuda.is_available() else "cpu"

eval_env = make_vec_env('MiniGrid-FourRooms-v1', 
                        n_envs=1, 
                        seed=0, 
                        vec_env_cls=DummyVecEnv, 
                        wrapper_class=gym_wrapper, 
                        env_kwargs={'agent_pos': train_config['agent positions'],
                                    'goal_pos': train_config['goal positions'],
                                    'doors_pos': train_config['topologies'],
                                    'agent_dir': train_config['agent directions']})

train_env = make_vec_env('MiniGrid-FourRooms-v1', 
                        n_envs=n_envs, 
                        seed=0, 
                        vec_env_cls=DummyVecEnv, 
                        wrapper_class=gym_wrapper, 
                        env_kwargs={'agent_pos': train_config['agent positions'],
                                    'goal_pos': train_config['goal positions'],
                                    'doors_pos': train_config['topologies'],
                                    'agent_dir': train_config['agent directions']})


policy_kwargs = dict(features_extractor_class = CNN, features_extractor_kwargs = {'features_dim': 512}, normalize_images=False, net_arch=[])

callback = EvalCallback(eval_env, n_eval_episodes=num_train_configs, eval_freq=max(10000 // n_envs, 1), verbose=0)
# look up CheckpointCallback if you want to store network checkpoints or replay buffers during training 
# https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#checkpointcallback

# Delete the following lines if you don't want to use wandb for logging results
import wandb
from wandb.integration.sb3 import WandbCallback
with wandb.init(
        project="four-room-project",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        name="DQN",
        ):
    wandb_callback = WandbCallback()

    model = DQN(
        'MlpPolicy',
        train_env, 
        learning_starts=batch_size,
        tensorboard_log="logging/", 
        policy_kwargs=policy_kwargs, 
        learning_rate=learning_rate, 
        buffer_size=buffer_size, 
        replay_buffer_class=ReplayBuffer,
        batch_size=batch_size, 
        tau=tau, gamma=gamma, 
        train_freq=(train_freq // n_envs, "step"), 
        gradient_steps=gradient_steps, 
        max_grad_norm=max_grad_norm, 
        target_update_interval=target_update_interval,
        exploration_final_eps=exploration_final_eps,
        exploration_fraction=exp_frac,
        seed=0,
        device=device,
        )

    model.learn(total_timesteps=500_000, callback=callback)
    train_env.close()
    eval_env.close()
