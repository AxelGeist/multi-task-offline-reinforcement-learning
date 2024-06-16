import  discrete_SAC as sac
import  discrete_SAC_BC as sac_bc
import discrete_BC as bc
import dataset_gen_optimal_policy
import dataset_gen_suboptimal_policy
import dataset_gen_mixed_policy
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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys


# Function to adjust color lightness
def adjust_color_lightness(color, amount=0.5):
    """
    Adjusts the lightness of a given color.
    `color` should be a matplotlib color string.
    `amount` > 1 makes the color lighter, < 1 makes it darker.
    """
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(*c)
    new_l = max(0, min(1, l * amount))
    r, g, b = colorsys.hls_to_rgb(h, new_l, s)
    return mcolors.to_hex((r, g, b))

# 0. Learning Curve of BC, SAC, SAC+BC in different Environments & Datasets -> 5 seeds
## BC on optimal dataset over 50k training steps 
## BC on Suboptimal dataset over 50k training steps
## BC on Mixed dataset over 50k training steps
## SAC on optimal dataset over 50k training steps
## SAC on Suboptimal dataset over 50k training steps
## SAC on Mixed dataset over 50k training steps
## SAC+BC on optimal dataset over 50k training steps
## SAC+BC on Suboptimal dataset over 50k training steps
## SAC+BC on Mixed dataset over 50k training steps

def plotLearningCurve(dataset_quality: str, algorithm: str):
    
    if dataset_quality == 'optimal':
        dataset_average = 1.0
        dataset_size = "40"
    elif dataset_quality == 'suboptimal':
        dataset_average = 0.5
        dataset_size = "80"
    elif dataset_quality == 'mixed':
        dataset_average = 1.0
        dataset_size = "80"
    
    df = merge_all_results(
        [
            f'models/{algorithm}/{dataset_quality}_{dataset_size}_10/results.csv', 
            f'models/{algorithm}/{dataset_quality}_{dataset_size}_11/results.csv', 
            f'models/{algorithm}/{dataset_quality}_{dataset_size}_12/results.csv', 
            f'models/{algorithm}/{dataset_quality}_{dataset_size}_13/results.csv', 
            f'models/{algorithm}/{dataset_quality}_{dataset_size}_14/results.csv', 
        ])
    

    
    print(df)
    
    # Grouping by Steps, Environment, and calculating mean reward and standard deviation for each step
    grouped_env = df.groupby(["Steps", "Environment"]).agg({"Reward_mean": "mean", "Reward_std": "mean"}).reset_index()



    # Draw standard deviation with lines
    # plt.figure(figsize=(10, 6))

    # for environment in grouped_env["Environment"].unique():
    #     env_data = grouped_env[grouped_env["Environment"] == environment]
    #     plt.errorbar(env_data["Steps"], env_data["Reward_mean"], yerr=env_data["Reward_std"], fmt='-o', capsize=5, label=environment)

    # plt.xlabel("Training Steps")
    # plt.ylabel("Mean Reward")
    # plt.title("Mean Reward vs. Training Steps by Environment (Averaged over 5 seeds) with Standard Deviation")
    # plt.legend(title="Environment")
    # plt.grid(True)
    # plt.show()
    
    
    # Draw standard deviation as a shaded area in the background
    plt.figure(figsize=(12, 8))

    for environment in grouped_env["Environment"].unique():
        env_data = grouped_env[grouped_env["Environment"] == environment]
        steps = env_data["Steps"].values
        reward_mean = env_data["Reward_mean"].values
        # reward_std = env_data["Reward_std"].values
        plt.plot(steps, reward_mean, marker='o', label=environment)
        # plt.fill_between(steps, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)

    dataset_line = plt.axhline(y=dataset_average, color='red', linestyle='dotted', linewidth=1, label=f'{dataset_quality} Dataset Average')
    dataset_line.set_dashes([5, 10])  # 5 points on, 10 points off

    plt.xlabel("Training Steps")
    plt.ylabel("Mean Reward")
    plt.title(f"{algorithm.upper()} Learning Curve - {dataset_quality.capitalize()} Dataset with {dataset_size} Transitions - Averaged over 5 seeds")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'results/learning_curve/{algorithm}_{dataset_quality}')
    # plt.show()






# 1. Generalization to new Environments -> use the data above in 25k steps (show deviation within the 5 seeds)
## 40x Optimal dataset - BC (Train, Test100, Test0) vs. SAC+BC (Train, Test100, Test0) vs. SAC (Train, Test100, Test0)
## 80x Suboptimal dataset - BC (Train, Test100, Test0) vs. SAC+BC (Train, Test100, Test0) vs. SAC (Train, Test100, Test0)
## 80x Mixed dataset - BC (Train, Test100, Test0) vs. SAC+BC (Train, Test100, Test0) vs. SAC (Train, Test100, Test0)

def plotGeneralizationGraph(dataset_quality: str, training_steps: int):
    # TODO: add 15 links (5 for each algorithms results, since 5 seeds)
        
    if dataset_quality == 'optimal':
        dataset_average = 1.0
        dataset_size = "40"
    elif dataset_quality == 'suboptimal':
        dataset_average = 0.5
        dataset_size = "80"
    elif dataset_quality == 'mixed':
        dataset_average = 1.0
        dataset_size = "80"
        
    df = merge_all_results(
    [
        f'models/bc/{dataset_quality}_{dataset_size}_10/results.csv', 
        f'models/bc/{dataset_quality}_{dataset_size}_11/results.csv', 
        f'models/bc/{dataset_quality}_{dataset_size}_12/results.csv', 
        f'models/bc/{dataset_quality}_{dataset_size}_13/results.csv', 
        f'models/bc/{dataset_quality}_{dataset_size}_14/results.csv', 

        f'models\sac_bc\{dataset_quality}_{dataset_size}_10/results.csv',
        f'models\sac_bc\{dataset_quality}_{dataset_size}_11/results.csv',
        f'models\sac_bc\{dataset_quality}_{dataset_size}_12/results.csv',
        f'models\sac_bc\{dataset_quality}_{dataset_size}_13/results.csv',
        f'models\sac_bc\{dataset_quality}_{dataset_size}_14/results.csv',

        f'models\sac\{dataset_quality}_{dataset_size}_10/results.csv',
        f'models\sac\{dataset_quality}_{dataset_size}_11/results.csv',
        f'models\sac\{dataset_quality}_{dataset_size}_12/results.csv',
        f'models\sac\{dataset_quality}_{dataset_size}_13/results.csv',
        f'models\sac\{dataset_quality}_{dataset_size}_14/results.csv',

    ])
    
    # Filter the data for steps
    df_step = df[df['Steps'] == training_steps]

    # Compute the average of Reward_mean for each algorithm
    reward_mean_avg_env = df_step.groupby(['Algorithm', 'Environment'])['Reward_mean'].mean()
    reward_mean_std = df_step.groupby(['Algorithm', 'Environment'])['Reward_mean'].std()

    # Convert the series to a DataFrame for easy plotting
    reward_mean_avg_env = reward_mean_avg_env.unstack()
    reward_mean_std = reward_mean_std.unstack()

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    reward_mean_avg_env.plot(kind='bar', yerr=reward_mean_std, capsize=4, ax=ax,
        color=['skyblue', 'lightgreen', 'salmon'])
    ax.set_title(f'{dataset_quality.capitalize()} Dataset with {dataset_size} Transitions - {int(training_steps / 1000)}k Training Steps')
    ax.set_ylabel('Reward Mean')
    ax.set_xlabel('')
    ax.set_xticks(range(len(reward_mean_avg_env.index)))
    ax.set_xticklabels(reward_mean_avg_env.index, rotation=0)
    ax.legend(title='Environment')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.005)
    
    ax.axhline(y=dataset_average, color='r', linestyle='--', linewidth=2, label=f'{dataset_quality.capitalize()} Dataset Average')
    ax.legend()  # Update legend to include the new line
    
    
    plt.savefig(f'results/generalization_{dataset_quality}')
    # plt.show()



# X. Data Diversity Graphs -> not possible, since I dont have enough training levels (hundreds of different datasets)
## Number of training levels X to X - SAC+BC (Train, Test100, Test0)
## Number of training levels X to X- BC (Train, Test100, Test0)


# 2. Training & Testing on the Same Environment

def plotTrainingEnvironmentGraph(training_steps: int):
    
    df = pd.DataFrame()

    for dataset_quality in ['optimal', 'suboptimal', 'mixed']:
        if dataset_quality == 'optimal':
            dataset_size = "40"
        elif dataset_quality == 'suboptimal':
            dataset_size = "80"
        elif dataset_quality == 'mixed':
            dataset_size = "80"
            
        for algorithm in ['bc', 'sac', 'sac_bc']:
                
            new_df = merge_all_results(
                [
                    f'models/{algorithm}/{dataset_quality}_{dataset_size}_10/results.csv', 
                    f'models/{algorithm}/{dataset_quality}_{dataset_size}_11/results.csv', 
                    f'models/{algorithm}/{dataset_quality}_{dataset_size}_12/results.csv', 
                    f'models/{algorithm}/{dataset_quality}_{dataset_size}_13/results.csv', 
                    f'models/{algorithm}/{dataset_quality}_{dataset_size}_14/results.csv', 
                ])
            
            df = pd.concat([df, new_df])

    # Filter the data for steps
    df_step = df[df['Steps'] == training_steps]

    # Filter the environment
    df_train = df_step[df_step['Environment'] == 'Train']

    # Compute the average and standard deviation of Reward_mean for each combination of Algorithm and Dataset
    reward_mean_avg_env = df_train.groupby(['Algorithm', 'Dataset'])['Reward_mean'].mean().unstack()
    reward_mean_std = df_train.groupby(['Algorithm', 'Dataset'])['Reward_mean'].std().unstack()

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    reward_mean_avg_env.T.plot(kind='bar', yerr=reward_mean_std.T, capsize=4, ax=ax,
        color=['skyblue', 'lightgreen', 'salmon'])

    ax.set_title(f'Training & Testing on the Same Environment - {int(training_steps / 1000)}k Training Steps')
    ax.set_ylabel('Reward Mean')
    ax.set_xlabel('')
    plt.xticks(rotation=0)
    ax.set_xticklabels([label.get_text().capitalize() for label in ax.get_xticklabels()])

    
    # ax.set_xticks(range(len(reward_mean_avg_env.index)))
    # ax.set_xticklabels(reward_mean_avg_env.index, rotation=0)
    ax.legend(title='Algorithm')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.005)

    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label=f'Optimal Dataset Average')
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label=f'Mixed Dataset Average')
    ax.axhline(y=0.5, color='blue', linestyle='--', linewidth=2, label=f'Suboptimal Dataset Average')
    ax.legend()  # Update legend to include the new line

    plt.savefig(f'results/training_environment/train_env_{training_steps}')
    # plt.show()







# 3. Dataset Size Graphs
## 40x to 400x Optimal dataset - SAC+BC (Train, Test100, Test0) vs. SAC (Train, Test100, Test0) vs. BC (Train, Test100, Test0)
## 80x to 400x Suboptimal dataset - SAC+BC (Train, Test100, Test0)
## 80x to 400x Suboptimal dataset - SAC (Train, Test100, Test0)
## 80x to 400x Suboptimal dataset - BC (Train, Test100, Test0)

def plotVariousDatasetSizeGraph(dataset_quality: str):
    
    ## dataset sizes: 40, 80, 200, 400  -> so need model trained & evaluated on each of them for only 20k training steps
    
    if dataset_quality == 'optimal':
        dataset_average = 1.0
        training_steps = 20000
    elif dataset_quality == 'suboptimal':
        dataset_average = 0.5
        training_steps = 20000
    elif dataset_quality == 'mixed':
        dataset_average = 1.0
        training_steps = 50000
    
    df = pd.DataFrame()
    
    for dataset_size in [40, 80, 200, 400]:
        for algorithm in ['bc', 'sac', 'sac_bc']:
        
            new_df = merge_all_results(
                [
                    f'models/{algorithm}/{dataset_quality}_{dataset_size}_10/results.csv', 
                    f'models/{algorithm}/{dataset_quality}_{dataset_size}_11/results.csv', 
                    f'models/{algorithm}/{dataset_quality}_{dataset_size}_12/results.csv', 
                    f'models/{algorithm}/{dataset_quality}_{dataset_size}_13/results.csv', 
                    f'models/{algorithm}/{dataset_quality}_{dataset_size}_14/results.csv', 
                ])
            
            df = pd.concat([df, new_df])
    
    print(df)
    
    # TODO: plot graphs
    # x axis -> dataset_size
    # y axis -> Mean Reward
    # line chart
    # only display the given dataset_quality 
    
    algorithm_base_colors = {
        'BC': 'red',
        'SAC': 'blue',
        'SAC+BC': 'green',
    }
    
    environment_lightness = {
        'Train': 1.5,  # lighter
        'Test_Reachable': 1.0,  # base color
        'Test_Unreachable': 0.5  # darker
    }
    
    df_filtered = df[df['Steps'] == training_steps]
    # Group by Size, Algorithm, and Environment, then calculate the mean of Reward_mean
    df_grouped_env = df_filtered.groupby(['Size', 'Algorithm', 'Environment']).agg({'Reward_mean': 'mean'}).reset_index()

    # Plot
    plt.rcParams.update({'font.size': 15})  # Adjust this value as needed
    plt.figure(figsize=(14, 10))
    for algorithm in df_grouped_env['Algorithm'].unique():
        for environment in df_grouped_env['Environment'].unique():
            subset = df_grouped_env[(df_grouped_env['Algorithm'] == algorithm) & (df_grouped_env['Environment'] == environment)]
            color = adjust_color_lightness(algorithm_base_colors[algorithm], environment_lightness[environment])
            plt.plot(subset['Size'], subset['Reward_mean'], marker='o', label=f'{algorithm} - {environment}', color=color)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.005)
    plt.axhline(y=dataset_average, color='purple', linestyle='--', linewidth=2, label=f'{dataset_quality.capitalize()} Dataset Average')    

    plt.xticks(df_grouped_env['Size'].unique())    
    plt.xlabel('Dataset Size (Episodes)')
    plt.ylabel('Reward Mean')
    plt.title(f'Increasing {dataset_quality.capitalize()} Dataset Size (40 to 400 Episodes) - {int(training_steps / 1000)}k Training Steps')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'results/dataset_size/size_{dataset_quality}')
    # plt.show()





# 4. How much overfitting on the training model performs the best
# which SAC-N model performs the best on test100_config and test0_config?
# use models from 0.1-1.0 mean reward on evaluating on training_config

############# Configuration ##########################################################

CONFIG = {
    "training_datasets": {
        # "optimal": "./datasets/expert_dataset_iql.pkl",
        "optimal": "./datasets/optimal_40x.pkl",
        "suboptimal": "./datasets/suboptimal_80x.pkl"
    },
    "evaluation_models": {
        "optimal": {
            "sac": "./models/sac/optimal_2999epochs.pt",
            "bc": "./models/bc/BC_model_optimal.d3",
        },
        "suboptimal": {
            "sac": "./models/sac/suboptimal_2999epochs.pt",
            "bc": "./models/bc/BC_model_suboptimal.d3"
        }
    }
}

############# Dataset Generation ######################################################

def generate_datasets():
    # TODO: implement and import dataset_gen_mixed_policy
    dataset_gen_optimal_policy.generate_dataset()
    dataset_gen_suboptimal_policy.generate_dataset()
    dataset_gen_mixed_policy.generate_dataset()

############# Training ###############################################################

def train_rl_models(dataset_name):
    dataset_path = CONFIG["training_datasets"][dataset_name]
    # sac.train((dataset_name, dataset_path))
    # sac_bc.train((dataset_name, dataset_path))
    bc.train((dataset_name, dataset_path))

############# Evaluation #############################################################

def evaluate_rl_models(model_type):
    model_paths = CONFIG["evaluation_models"][model_type]
    sac_eval_data = sac.eval(model_paths=model_paths) # Works!
    sac_bc_eval_data = sac_bc.eval(model_paths=model_paths) # Works!
    bc_eval_data = bc.eval(model_paths=model_paths)
    return bc_eval_data, sac_bc_eval_data, sac_eval_data


############# Merge All Results #############################################################

def merge_all_results(result_csv_paths: List[str]):
    
    merged_df = pd.DataFrame()
    
    for csv_path in result_csv_paths:
        df = pd.read_csv(csv_path)
        merged_df = pd.concat([merged_df, df])
        
        
    # Rename environments
    merged_df['Environment'] = merged_df['Environment'].replace('train', 'Train')
    merged_df['Environment'] = merged_df['Environment'].replace('test_100', 'Test_Reachable')
    merged_df['Environment'] = merged_df['Environment'].replace('test_0', 'Test_Unreachable')
    
    # Order from Train to Test_Unreachable
    merged_df['Environment'] = pd.Categorical(merged_df['Environment'], categories=['Train', 'Test_Reachable', 'Test_Unreachable'], ordered=True)
    

    # print(merged_df)
    return merged_df




############# Plotting ################################################################

def plot_results(bc_eval_data: pd.DataFrame, sac_bc_eval_data: pd.DataFrame, sac_eval_data: pd.DataFrame, ):
    results = pd.concat([bc_eval_data, sac_bc_eval_data, sac_eval_data])
    print(results)
    # Plotting
    plt.figure(figsize=(10, 9))
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
    training_dataset_quality = "optimal"  # "optimal" or "suboptimal"
    # train_rl_models(training_dataset_quality)

    # 3. Choose models for evaluation & plotting
    evaluation_model_quality = "optimal"  # "optimal" or "suboptimal"
    # bc_eval_data, sac_bc_eval_data, sac_eval_data = evaluate_rl_models(evaluation_model_quality)
    # TODO: store those results somehwere?!

    # sac_bc_eval_data = pd.DataFrame({
    #     'Algorithm': 'SAC+BC',
    #     'Environment': ['train', 'test_100', 'test_0'],
    #     'Reward_mean': [1.0, 0.4, 0.3],
    #     'Reward_std': [0.000000, 0.263391, 0.300000]
    # })

    # sac_eval_data = pd.DataFrame({
    #     'Algorithm': 'SAC',
    #     'Environment': ['train', 'test_100', 'test_0'],
    #     'Reward_mean': [0.125, 0.075, 0.100],
    #     'Reward_std': [0.000000, 0.263391, 0.300000]
    # })

    # bc_eval_data = pd.DataFrame({
    #     'Algorithm': 'BC',
    #     'Environment': ['train', 'test_100', 'test_0'],
    #     'Reward_mean': [1.0, 0.4, 0.3],
    #     'Reward_std': [0.0, 0.0, 0.0]
    # })
    
    # 4. Plot results
    # TODO: fetch the data from the results folder instead
    # plot_results(bc_eval_data, sac_bc_eval_data, sac_eval_data, )
    
    
    # results
    # plotLearningCurve(dataset_quality='mixed', algorithm = 'sac_bc')    
    # plotGeneralizationGraph(dataset_quality = 'mixed', training_steps = 50000)
    # plotTrainingEnvironmentGraph(training_steps = 20000)
    plotVariousDatasetSizeGraph(dataset_quality = 'mixed')
    plotVariousDatasetSizeGraph(dataset_quality = 'optimal')
    plotVariousDatasetSizeGraph(dataset_quality = 'suboptimal')

    

if __name__ == "__main__":
    main()
    
    
    
    
