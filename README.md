# Experimental Evaluation of the Generalizability of the Soft Actor-Critic + Behavioral Cloning Algorithm

This repository is part of the Research Project CSE3000 in 2024 @ TU Delft. Other projects can be found [here](https://cse3000-research-project.github.io/2024/Q2).

The research questions were:
1. Can SAC combined with BC effectively generalize to new tasks within a multi-task reinforcement learning environment?
2. What characteristics of the offline dataset are critical for the success or failure of SAC+BC in such settings?

## Getting Started

### 1. Install Dependencies
```
pip install requirements -r requirements.txt
```

### 2. Open ./results.py
Follow Instructions in the `main()` Function in `results.py`

## Environment:
The environment used is the MiniGrid Four-Room which contain 3 configurations, of which one is the training configuration the other are testing configuration for reachable and unreachable tasks. Every configuration will contain 40 tasks. 

## Datasets
The datasets have been created with an Optimal, Suboptimal and Mixed policy.

### Optimal Policy - 100% success rate
![rendered_episode](https://github.com/AxelGeist/multi-task-offline-reinforcement-learning/assets/73136957/0d6a7199-8e6a-4484-9e87-de3e49ec4aa5)

### Suboptimal Policy - 50% success rate (learned from a Deep Q-Network)
![rendered_episode](https://github.com/AxelGeist/multi-task-offline-reinforcement-learning/assets/73136957/078642f3-e2d1-4628-b989-8e5db2d0214f)

### Mixed Optimal-Suboptimal Policy - 100% success rate
![mixed_policy](https://github.com/AxelGeist/multi-task-offline-reinforcement-learning/assets/73136957/eefba3be-40dd-4797-9fae-167421c47abd)


