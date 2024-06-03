import optuna
import d3rlpy
import argparse
import logging
import dill
import numpy as np
from d3rlpy.dataset import MDPDataset
import discrete_BC

def initialize_train_env():
    return discrete_BC.initialize_envs()["train"]

def get_config():
    return discrete_BC.Config

def objective(trial):
    dataset_path = "./datasets/dataset_gen_suboptimal_policy_50pct_80x.pkl"
    
    # Define the hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_steps = trial.suggest_int('n_steps', 40, 400)

    # Load your dataset
    with open(dataset_path, 'rb') as file:
        d4rl_dataset = dill.load(file)
        
    mdp_dataset = MDPDataset(
        observations=d4rl_dataset['observations'],
        actions=d4rl_dataset['actions'],
        rewards=d4rl_dataset['rewards'],
        # next_observations=d4rl_dataset['next_observations'],
        terminals=d4rl_dataset['terminals']
    )
    
    # Setup the BC model with the trial's current suggestions
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    d3rlpy.seed(args.seed)
    bc_config = d3rlpy.algos.DiscreteBCConfig(learning_rate=learning_rate)
    bc = bc_config.create(device=args.gpu)

    # Assuming `dataset` is already loaded and split into features and targets
    bc.fit(
        mdp_dataset, 
        n_steps=n_steps, 
        n_steps_per_epoch=n_steps, 
        experiment_name="Tuning_BC"
    )
    
    # Evaluate the model performance
    train_env = initialize_train_env()
    config = get_config()
    rewards = discrete_BC.evaluate_env(train_env, bc, config, "train")
    print("REWARDS:", rewards)
    performance = np.mean(rewards)

    return performance  # Optuna tries to minimize this value by default


from optuna.visualization import plot_optimization_history, plot_contour, plot_rank

def optimize_bc():
    study_name = "tuning_bc"  # Unique identifier of the study.
    storage_name = "sqlite:///tuning_bc.db".format(study_name)
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, load_if_exists=True)
    # study = optuna.create_study(direction='maximize', study_name='bc_tuning')  # Use 'minimize' for loss, 'maximize' for accuracy or other performance metrics
    # study.optimize(objective, n_trials=1)  # Number of trials to perform

    print("Best hyperparameters: ", study.best_trial.params) 
    print("Best value: ", study.best_trial.value)    

    fig = plot_optimization_history(study)
    fig.update_layout(title="History: BC Tuning on 80x Suboptimal (50%) Dataset",)
    fig.write_image(f"results/{study_name}_history.png") 
    fig = plot_rank(study)
    fig.update_layout(title="Rank: BC Tuning on 80x Suboptimal (50%) Dataset",)
    fig.write_image(f"results/{study_name}_rank.png")
    fig = plot_contour(study, params=[ "learning_rate", "n_steps"])
    fig.update_layout(title="Contour: BC Tuning on 80x Suboptimal (50%) Dataset",)
    fig.write_image(f"results/{study_name}_contour.png") 

    return study.best_trial.params


if __name__ == "__main__":
    optimize_bc()

