import gymnasium as gym
import dill
import pickle
import numpy as np
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper
from four_room.shortest_path import find_all_action_values
from four_room.utils import obs_to_state

# Create and register a custom environment
gym.register('MiniGrid-FourRooms-v1', entry_point=FourRoomsEnv)

with open('four_room/configs/fourrooms_train_config.pl', 'rb') as file: train_config = dill.load(file)

env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1', 
    agent_pos=train_config['agent positions'],
    goal_pos=train_config['goal positions'],
    doors_pos=train_config['topologies'],
    agent_dir=train_config['agent directions']))


class Agent:
    # def __init__(self):
    #     pass

    ## Optimal Policy
    def decide_action(self, obs):
        ## return Optimal Policy in a given state 'obs'
        state = obs_to_state(obs)
        q_values = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99)
        optimal_action = np.argmax(q_values)
        return optimal_action
        

    # def learn(self, state, action, reward, next_state, done):
    #     pass


# Instantiate the agent
agent = Agent()

num_episodes = 100
dataset = [] # replay buffer in the D4RL format

for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.decide_action(obs)  
        next_obs, reward, done, truncated, info = env.step(action)

        # agent.learn(obs, action, reward, next_obs, done)
        dataset.append((obs, action, reward, next_obs, done))

        obs = next_obs
        total_reward += reward
    
    print(f'Episode {episode + 1}: Total Reward = {total_reward}')


# Save the dataset to a file
with open('four_room_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)



env.close()
