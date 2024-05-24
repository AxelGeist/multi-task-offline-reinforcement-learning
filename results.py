import  discrete_SAC_N as sac_n
import discrete_BC as bc
import dataset_gen_optimal_policy
import dataset_gen_suboptimal_policy
import math
import os
import random
import uuid
from dataclasses import dataclass, field
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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import dill
import pickle
from torch.distributions import Normal
from tqdm import trange
import argparse
import d3rlpy
from d3rlpy.algos import DiscreteBCConfig, DiscreteBC
from d3rlpy.metrics import TDErrorEvaluator
from d3rlpy.metrics import EnvironmentEvaluator
from d3rlpy.dataset import MDPDataset

# 1. Generalization to new Environments -> Midterm
## 40x Expert dataset - SAC-N+BC (Train, Test100, Test0) vs. BC (Train, Test100, Test0)
## 80x Suboptimal dataset - SAC-N+BC (Train, Test100, Test0) vs. BC (Train, Test100, Test0)

# 2. Data Diversity Graphs -> not possible, since I dont have enough training levels (hundreds of different datasets)
## Number of training levels X to X - SAc-N+BC (Train, Test100, Test0)
## Number of training levels X to X- BC (Train, Test100, Test0)

# 3. Data Size Graphs
## 40x to 400x Expert dataset - SAC-N+BC (Train, Test100, Test0) vs. BC (Train, Test100, Test0)
## 40x to 400x Suboptimal dataset - SAC-N+BC (Train, Test100, Test0)
## 40x to 400x Suboptimal dataset - BC (Train, Test100, Test0)

# 4. How much overfitting on the training model performs the best
# which SAC-N model performs the best on test100_config and test0_config?
# use models from 0.1-1.0 mean reward on evaluating on training_config

############# Configuration ##########################################################

CONFIG = {
    "training_datasets": {
        "optimal": "./datasets/dataset_gen_optimal_policy_40x.pkl",
        "suboptimal": "./datasets/dataset_gen_suboptimal_policy_50pct_80x.pkl"
    },
    "evaluation_models": {
        "optimal": {
            "sac": "./models/sac-n/optimal_2999epochs.pt",
            "bc": "./models/bc/BC_model_optimal.d3",
        },
        "suboptimal": {
            "sac": "./models/sac-n/suboptimal_2999epochs.pt",
            "bc": "./models/bc/BC_model_suboptimal.d3"
        }
    }
}

############# Dataset Generation ######################################################

def generate_datasets():
    # TODO: implement and import dataset_gen_mixed_policy
    dataset_gen_optimal_policy.generate_dataset()
    dataset_gen_suboptimal_policy.generate_dataset()

############# Training ###############################################################

def train_rl_models(dataset_name):
    dataset_path = CONFIG["training_datasets"][dataset_name]
    sac_n.train((dataset_name, dataset_path))
    bc.train((dataset_name, dataset_path)) # TODO: merge eval & train bc first

############# Evaluation #############################################################

def evaluate_rl_models(model_type):
    model_paths = CONFIG["evaluation_models"][model_type]
    sac_eval_data = sac_n.eval(model_paths=model_paths) # Works!
    bc_eval_data = bc.eval(model_paths=model_paths)
    return sac_eval_data, bc_eval_data

############# Plotting ################################################################

def plot_results(sac_eval_data: pd.DataFrame, bc_eval_data: pd.DataFrame):
    results = pd.concat([sac_eval_data, bc_eval_data])
    
    # Plotting
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x="Algorithm", y="Reward_mean", hue="Environment", data=results, errorbar="sd", palette="muted")
    
    # Add Dataset Line
    target_line = plt.axhline(1.0, color='grey', linestyle='--', linewidth=2)
    handles, labels = barplot.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='grey', lw=2, linestyle='--'))
    labels.append('Optimal Dataset Average')
    plt.legend(handles, labels, title='')    
        
    plt.title("Four_room - 40x Optimal Dataset")
    plt.ylabel("Reward_mean")
    plt.xlabel("")
    plt.show()

############# Main Execution #########################################################

def main():
    # 1. Generate datasets
    # generate_datasets()
    
    # 2. Choose dataset for training
    # training_dataset_quality = "suboptimal"  # "optimal" or "suboptimal"
    # train_rl_models(training_dataset_quality)

    # 3. Choose models for evaluation & plotting
    evaluation_model_quality = "optimal"  # "optimal" or "suboptimal"
    sac_eval_data, bc_eval_data = evaluate_rl_models(evaluation_model_quality)
    # TODO: store those results somehwere?!

    # sac_eval_data = pd.DataFrame({
    #     'Environment': ['train', 'test_100', 'test_0'],
    #     'Reward_mean': [0.500, 0.075, 0.100],
    #     'Reward_std': [0.000000, 0.263391, 0.300000]
    # })

    # bc_eval_data = pd.DataFrame({
    #     'Environment': ['train', 'test_100', 'test_0'],
    #     'Reward_mean': [1.0, 0.4, 0.3],
    #     'Reward_std': [0.0, 0.0, 0.0]
    # })

    # sac_eval_data['Algorithm'] = 'SAC'
    # bc_eval_data['Algorithm'] = 'BC'
    
    # 4. Plot results
    # TODO: fetch the data from the results folder instead
    plot_results(sac_eval_data, bc_eval_data)
    

if __name__ == "__main__":
    main()
