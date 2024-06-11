from env import FourRoomsEnv
from wrappers import gym_wrapper
import imageio
import numpy as np
from pyvirtualdisplay import Display  
import dill
import gymnasium as gym

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open('configs/fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)

env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1', 
                                                    agent_pos= train_config['agent positions'],
                                                    goal_pos = train_config['goal positions'],
                                                    doors_pos = train_config['topologies'],
                                                    agent_dir = train_config['agent directions'],
                                                    render_mode="rgb_array"))

with Display(visible=False) as disp:
    images = []
    for i in range(len(train_config['topologies'])):
        obs, _ = env.reset()
        img = env.render()
        done = False
        while not done:
            images.append(img)
            # retrieve your action here
            action = env.action_space.sample() # for example, just sample a random action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            img = env.render()

                


    imageio.mimsave('rendered_episode.gif', [np.array(img) for i, img in enumerate(images) if i%1 == 0], duration=200)
