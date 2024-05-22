import  discrete_SAC_N as sac_n
import discrete_BC as bc
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




############# Training ###############################################################
# TODO: choose the offline dataset to be used for training the RL algorithm
# dataset_tuple = ("optimal", "./datasets/dataset_gen_optimal_policy_40x.pkl")
dataset_tuple = ("suboptimal", "./datasets/dataset_gen_suboptimal_policy_50pct_80x.pkl")

sac_n.train(dataset_tuple)
bc.train(dataset_tuple) # TODO: merge eval & train bc first





############# Evaluation #############################################################

# need the best trained model of SAC-N?! Or which model should I use? 

# TODO: choose the trained models to be used for evaluating the RL algorithm
model_paths = {
    "sac": "./models/sac-n/optimal_2999epochs.pt",
    "bc": "./models/bc/BC_model_optimal.d3",
}
# model_paths = {
#     "sac": "./models/sac-n/suboptimal_2999epochs.pt",
#     "bc": "./models/bc/BC_model_suboptimal.d3"
# }

sac_eval_data = sac_n.eval(model_paths=model_paths) # Works!
bc_eval_data = bc.eval(model_paths=model_paths)


# TODO: plot results with the evaluation data 
















