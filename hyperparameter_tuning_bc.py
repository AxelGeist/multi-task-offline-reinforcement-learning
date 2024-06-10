import optuna
import d3rlpy
import argparse
import logging
import dill
import numpy as np
from d3rlpy.dataset import MDPDataset
import discrete_BC

# Specify dataset
dataset_quality = "optimal"
dataset_size = "40"
dataset_path = f"./datasets/{dataset_quality}_{dataset_size}x.pkl"


def objective(trial):
    
    config = discrete_BC.Config

    
    # Define the hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 3e-4)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 100, 128])
    n_steps = trial.suggest_int('n_steps', 50, 1000, 50)

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
    
    performance_list = []
    
    for seed in range(5):
    
        # Setup the BC model with the trial's current suggestions
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=seed)
        parser.add_argument("--gpu", type=int)
        args = parser.parse_args()
        d3rlpy.seed(args.seed)
        bc_config = d3rlpy.algos.DiscreteBCConfig(learning_rate=learning_rate, batch_size=batch_size)
        bc = bc_config.create(device=args.gpu)

        # Assuming `dataset` is already loaded and split into features and targets
        bc.fit(
            mdp_dataset, 
            n_steps=n_steps, 
            n_steps_per_epoch=n_steps, 
            experiment_name="Tuning_BC"
        )
        
        # Evaluate the model performance
        train_env = discrete_BC.initialize_envs()["train"]
        rewards = discrete_BC.evaluate_env(train_env, bc, config, "train")
        performance = np.mean(rewards)
        performance_list.append(performance)
        print("performance_list:", performance_list)
        
    average_performance = np.mean(performance_list)

    return average_performance  # Optuna tries to minimize this value by default


from optuna.visualization import plot_optimization_history, plot_contour, plot_rank

def optimize_bc():
    study_name = "tuning_bc"  # Unique identifier of the study.
    storage_name = f"sqlite:///{study_name}_{dataset_quality}.db".format(study_name)
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, load_if_exists=True)
    # study = optuna.create_study(direction='maximize', study_name='bc_tuning')  # Use 'minimize' for loss, 'maximize' for accuracy or other performance metrics
    # study.optimize(objective, n_trials=50)  # Number of trials to perform


    print("Best hyperparameters: ", study.best_trial.params) 
    print("Best value: ", study.best_trial.value)    

    fig = plot_optimization_history(study)
    fig.update_layout(title=f"History: BC Tuning on {dataset_size}x {dataset_quality.capitalize()} Dataset",)
    fig.write_image(f"results/tuning/{study_name}_{dataset_quality}_history.png") 
    fig = plot_rank(study)
    fig.update_layout(title=f"Rank: BC Tuning on {dataset_size}x {dataset_quality.capitalize()} Dataset",)
    fig.write_image(f"results/tuning/{study_name}_{dataset_quality}_rank.png")
    fig = plot_contour(study, params=[ "learning_rate", "n_steps"])
    fig.update_layout(title=f"Contour: BC Tuning on {dataset_size}x {dataset_quality.capitalize()} Dataset",)
    fig.write_image(f"results/tuning/{study_name}_{dataset_quality}_contour.png") 

    return study.best_trial.params


if __name__ == "__main__":
    optimize_bc()

