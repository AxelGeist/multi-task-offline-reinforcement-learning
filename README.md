# Experimental Evaluation of the Generalizability of the Soft Actor-Critic + Behavioral Cloning Algorithm

This GitHub repository contains the code, datasets, models and results of the research project CSE3000 @ TU Delft. The primary focus of this research was to evaluate the generalization capabilities of the Soft Actor-Critic (SAC) algorithm when combined with Behavioral Cloning (BC). 

The research questions were:
1. Can SAC combined with BC effectively generalize to new tasks within a multi-task reinforcement learning environment?
2. What characteristics of the offline dataset are critical for the success or failure of SAC+BC in such settings?

## Getting Started

### 1. Install Dependencies
```
pip install requirements -r requirements.txt
```

### 2. Create Datasets

## Environment:
The environment used is the MiniGrid Four-Room which contain 3 configurations, of which one is the training configuration the other are testing configuration for reachable and unreachable tasks. Every configuration will contain 40 tasks. 

## Datasets
The datasets have been created with an Optimal, Suboptimal and Mixed policy.

### Optimal Policy - 100% success rate
![rendered_episode](https://github.com/AxelGeist/multi-task-offline-reinforcement-learning/assets/73136957/0d6a7199-8e6a-4484-9e87-de3e49ec4aa5)

### Suboptimal Policy - 50% success rate (learned from a DQN Deep Q-Network)
![rendered_episode](https://github.com/AxelGeist/multi-task-offline-reinforcement-learning/assets/73136957/078642f3-e2d1-4628-b989-8e5db2d0214f)

### Mixed Optimal-Suboptimal Policy - 100% success rate
![mixed_policy](https://github.com/AxelGeist/multi-task-offline-reinforcement-learning/assets/73136957/eefba3be-40dd-4797-9fae-167421c47abd)


