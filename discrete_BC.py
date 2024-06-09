import math
import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import imageio
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
import dill
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.distributions import Normal
from tqdm import trange
import argparse
import d3rlpy
from d3rlpy.algos import DiscreteBCConfig, DiscreteBC
from d3rlpy.metrics import TDErrorEvaluator
from d3rlpy.metrics import EnvironmentEvaluator
from d3rlpy.dataset import MDPDataset

from os import sys
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper


@dataclass
class Config:
    # wandb params
    project: str = "BC"
    group: str = "BC"
    name: str = "BC"
    # training params
    env_name: str = "MiniGrid-FourRooms-v1"
    n_steps: int = 3000 # TODO: 50k
    n_steps_per_epoch: int = 500 # TODO: 1k
    # evaluation params
    eval_episodes: int = 40
    # eval_every: int = 40
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 1
    eval_seed: int = 11 # TODO: should differ from train_seed
    log_every: int = 100
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

############# Environment #############################################################

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def initialize_envs():
    gym.register(Config.env_name, FourRoomsEnv)

    with open('four_room/configs/fourrooms_train_config.pl', 'rb') as file1: 
        train_config = dill.load(file1)
    with open('./four_room/configs/fourrooms_test_100_config.pl', 'rb') as file2:
        test_100_config = dill.load(file2)
    with open('four_room/configs/fourrooms_test_0_config.pl', 'rb') as file3: 
        test_0_config = dill.load(file3)

    train_env = wrap_env(gym_wrapper(gym.make(Config.env_name, 
        agent_pos=train_config['agent positions'],
        goal_pos=train_config['goal positions'],
        doors_pos=train_config['topologies'],
        agent_dir=train_config['agent directions'],
        render_mode="rgb_array")))
        
    eval_100_env = wrap_env(gym_wrapper(gym.make(Config.env_name, 
        agent_pos=test_100_config['agent positions'],
        goal_pos=test_100_config['goal positions'],
        doors_pos=test_100_config['topologies'],
        agent_dir=test_100_config['agent directions'],
        render_mode="rgb_array")))

    eval_0_env = wrap_env(gym_wrapper(gym.make(Config.env_name, 
        agent_pos=test_0_config['agent positions'],
        goal_pos=test_0_config['goal positions'],
        doors_pos=test_0_config['topologies'],
        agent_dir=test_0_config['agent directions'],
        render_mode="rgb_array")))
    
    return {
        "train": train_env,
        "test_100": eval_100_env,
        "test_0": eval_0_env
    }


############# TRAINING ################################################################

@pyrallis.wrap()
def train(config: Config, dataset_tuple: tuple):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.train_seed)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    
    dataset_optimality = dataset_tuple[0]
    dataset_path = dataset_tuple[1]
    
    with open(dataset_path, 'rb') as file:
        d4rl_dataset = dill.load(file)

    mdp_dataset = MDPDataset(
        observations=d4rl_dataset['observations'],
        actions=d4rl_dataset['actions'],
        rewards=d4rl_dataset['rewards'],
        # next_observations=d4rl_dataset['next_observations'],
        terminals=d4rl_dataset['terminals']
    )
    print("SIZE:", mdp_dataset.size())


    d3rlpy.seed(args.seed)
    bc = d3rlpy.algos.DiscreteBCConfig().create(device=args.gpu)

    bc.fit(
        mdp_dataset,
        n_steps=config.n_steps,
        n_steps_per_epoch=config.n_steps_per_epoch, # for monitoring purposes,
        experiment_name=f'{dataset_optimality}_{mdp_dataset.size()}_{config.train_seed}'
    )

    # bc.save(f'./models/bc/BC_model_{dataset_optimality}.d3')


############# EVALUATION ##############################################################

def evaluate_env(env, bc, config, env_name):
    rewards = []
    # images = []
    
    for episode in range(config.eval_episodes):
        obs, _ = env.reset(seed=config.eval_seed)
        # obs, _ = env.reset()
        # img = env.render()
        done = False
        steps = 0

        while not done:
            steps += 1
            # images.append(img)
            # action = bc.predict(obs.reshape(1, -1))[0]
            action = bc.predict(obs.flatten()[None, :])
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated 

            obs = next_obs
            # img = env.render()
        
        rewards.append(reward)
        # print(f'Episode {episode + 1} in {env_name}: Reward = {rewards}')
        
    # visualize the agents actions in the maze
    # gif_path = f'rendered_BC_{env_name}.gif'
    # imageio.mimsave(gif_path, [np.array(img) for i, img in enumerate(images) if i % 1 == 0], duration=200)
    # print(f"Saved GIF for {env_name} at {gif_path}") 
    return rewards

@pyrallis.wrap()
def eval(config: Config, model_path: str):
    # model_path = model_paths["bc"]
    # model_path = "d3rlpy_logs\DiscreteBC_20240530010539\model_400.d3"
    
    bc = d3rlpy.load_learnable(model_path)
    
    all_rewards = []
    env_list = initialize_envs()
    
    for env_name, eval_env in env_list.items():
        rewards = evaluate_env(eval_env, bc, config, env_name)
        all_rewards.append({
            "Algorithm": "BC", 
            f"Environment": env_name, 
            "Reward_mean": np.mean(rewards),
            "Reward_std": np.std(rewards)
        })
    
    df_rewards = pd.DataFrame(all_rewards)
    
    # Plotting the rewards
    # plot_rewards(df_rewards)
    
    return df_rewards

@pyrallis.wrap()
def eval_all_models(config: Config, model_dir: str):
    
    if 'suboptimal' in model_dir:
        dataset_quality = 'suboptimal'
    elif 'optimal' in model_dir:
        dataset_quality = 'optimal'
    elif 'mixed' in model_dir:
        dataset_quality = 'mixed'
    
    log_file_path = f'{model_dir}/results.csv'
    
    # Prepare environments
    env_list = initialize_envs()
    
    # Check if the log file exists, if not, open it to write with headers
    file_exists = False
    
    for x in range(int(config.n_steps / config.n_steps_per_epoch)):
        
        # load model
        current_steps = x * config.n_steps_per_epoch + config.n_steps_per_epoch
        model_path = f'{model_dir}/model_{current_steps}.d3'
        bc = d3rlpy.load_learnable(model_path)
        
        # evaluate
        all_rewards = []
        
        eval_seed = int(model_dir[-1:]) + 10
    
        for env_name, eval_env in env_list.items():
            rewards = evaluate_env(eval_env, bc, config, env_name)
            all_rewards.append({
                "Algorithm": "BC", 
                "Dataset": dataset_quality,
                "Environment": env_name, 
                "Reward_mean": np.mean(rewards),
                "Reward_std": np.std(rewards),
                "Steps": current_steps,
                "Seed": eval_seed,
            })
    
        df_rewards = pd.DataFrame(all_rewards)
        print(df_rewards)
        
        # log results
        df_rewards.to_csv(log_file_path, mode='a', header=not file_exists, index=False)
        # Ensure header is not written again
        if not file_exists:
            file_exists = True
    # plot
    # plot_evaluation_results(log_file_path)

def plot_evaluation_results(csv_file_path):
    # Read the accumulated results
    df = pd.read_csv(csv_file_path)
    # df.columns = ['Algorithm', 'Environment', 'Reward_mean', 'Reward_std', 'Steps']

    # Plotting
    plt.figure(figsize=(10, 6))
    environments = df['Environment'].unique()
    for env_name in environments:
        env_data = df[df['Environment'] == env_name]
        plt.errorbar(env_data['Steps'], env_data['Reward_mean'], label=env_name, fmt='-o')
        # plt.errorbar(env_data['Steps'], env_data['Reward_mean'], yerr=env_data['Reward_std'], label=env_name, fmt='-o')

    plt.xlabel('Training Steps')
    plt.ylabel('Mean Reward')
    plt.title('Performance of BC across Different Environments')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
def plot_rewards(df_rewards):

    # Plotting the rewards using seaborn
    plt.figure(figsize=(10, 6))
    # sns.barplot(data=df_rewards, x='Environment', y='Reward', hue='Optimality')
    sns.barplot(data=df_rewards, x='Algorithm', y='Reward', hue='Environment')
    plt.title('Mean Reward by Environment and Dataset Policy Optimality')
    plt.ylabel('Mean Reward')
    plt.xlabel('Environment')
    plt.legend(title='Optimality')
    plt.savefig('rewards_by_environment_and_optimality.png')  # Saves the plot into a file
    plt.show()



if __name__ == "__main__":    
    # train(("optimal", "./datasets/optimal_40x.pkl"))
    # train(("suboptimal", "./datasets/suboptimal_80x.pkl"))
    # train(("mixed", "./datasets/mixed_80x.pkl"))
    
    # eval(model_path="d3rlpy_logs\optimal_40_1_20240608223223\model_1000.d3")


    eval_all_models(model_dir="d3rlpy_logs\mixed_80_1_20240609033529")
    
    # eval_all_models(model_dir="models/bc/optimal_1")
    # eval_all_models(model_dir="models/bc/optimal_2")
    # eval_all_models(model_dir="models/bc/suboptimal_1")
    # eval_all_models(model_dir="models/bc/suboptimal_2")

    # plot_evaluation_results("d3rlpy_logs\DiscreteBC_20240606232020/results.csv")
    # plot_evaluation_results("d3rlpy_logs\DiscreteBC_20240530172702/results.csv")
