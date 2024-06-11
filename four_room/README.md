# 4-room
A simple 4-room grid world environment to test generalisation behaviour of RL agents. 

Requires Minigrid:
```
pip install minigrid
```
NetworkX
```
pip install networkx
```
and Dill
```
pip install dill
```

## Usage
From the parent directory of this repository you can import the environment with:
```
import gymnasium as gym
import dill
from four_room.env import FourRoomsEnv
from four_room.env_wrappers import gym_wrapper

gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open('four_room/configs/fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)

env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1', 
                                agent_pos=train_config['agent positions'], 
                                goal_pos=train_config['goal positions'], 
                                doors_pos=train_config['topologies'], 
                                agent_dir=train_config['agent directions']))
```
Similarly for the reachable testing config (```'four_room/configs/fourrooms_test_100_config.pl'```) and the unreachable testing config (```'four_room/configs/fourrooms_test_0_config.pl'```).

Utility functions for turning observations into state or images can be found in ```utils.py``` and code for finding the optimal trajectories/q-values for any state can be found in ```shortest_path.py```.

### Train DQN Agent
Example code for running a [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) DQN agent on the environment can be found in ```train_DQN.py```. In order to run it you additionally need to install:
```
pip install stablebaselines3 tensorboard wandb
```

### Optimal Actions
Use
```
from four_room.shortest_path import find_all_action_values
from four_room.utils import obs_to_state

state = obs_to_state(obs)
q_values = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99)
optimal_action = np.argmax(q_values)
```
to find the optimal action in a given state ```obs```.

### Rendering Episodes
Example code for rendering episodes can be found in ```render_episode.py```. You will need to additionally install:
```
pip install imageio pyvirtualdisplay
```
(```pyvirtualdisplay``` is required to render states on headless servers.) 
