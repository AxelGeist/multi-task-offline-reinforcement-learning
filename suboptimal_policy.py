import zipfile
import os
import gymnasium as gym
import dill
import pickle
import datetime
import numpy as np
import imageio
from stable_baselines3 import DQN
from pyvirtualdisplay import Display
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper
from four_room.shortest_path import find_all_action_values
from four_room.utils import obs_to_state

# load model -> TODO: choose which suboptimal policy you want
zip_path = './four_room/checkpoints/DQN_250000_steps.zip'
model = DQN.load(zip_path)

# Create and register a custom environment
gym.register('MiniGrid-FourRooms-v1', entry_point=FourRoomsEnv)

# TODO: Use the following to create offline datasets.
mazeConfig = 'four_room/configs/fourrooms_train_config.pl'

# TODO: Use the following to test TD3+BC and BC.
# mazeConfig = 'four_room/configs/fourrooms_test_100_config.pl'
# mazeConfig = 'four_room/configs/fourrooms_test_0_config.pl'

with open(mazeConfig, 'rb') as file: 
    train_config = dill.load(file)

env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1', 
    agent_pos=train_config['agent positions'],
    goal_pos=train_config['goal positions'],
    doors_pos=train_config['topologies'],
    agent_dir=train_config['agent directions'],
    render_mode="rgb_array"))

num_episodes = 100
dataset = { # replay buffer in the D4RL format
    'observations': [],
    'actions': [],
    'rewards': [],
    'next_observations': [],
    'terminals': [],
}

# with Display(visible=False) as disp:
images = []
for episode in range(num_episodes):
    obs, info = env.reset()
    img = env.render()
    done = False
    total_reward = 0

    while not done:
        images.append(img)
        action, _= model.predict(obs) 
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        dataset['observations'].append(obs)
        dataset['actions'].append(action)
        dataset['rewards'].append(reward)
        dataset['next_observations'].append(next_obs)
        dataset['terminals'].append(done)

        obs = next_obs
        total_reward += reward
        img = env.render()
    
    print(f'Episode {episode + 1}: Total Reward = {total_reward}')

# visualize the agents actions in the maze
imageio.mimsave('rendered_episode.gif', [np.array(img) for i, img in enumerate(images) if i%1 == 0], duration=200)

# Transform dataset arrays to np arrays, like in the DR4L format
for key in dataset:
    dataset[key] = np.array(dataset[key])

# Save the dataset to a file
policy = os.path.basename(__file__)[:-3]
date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'datasets/{policy}_{num_episodes}x_{date_time}.pkl'
with open(filename, 'wb') as f:
    pickle.dump(dataset, f)

env.close()