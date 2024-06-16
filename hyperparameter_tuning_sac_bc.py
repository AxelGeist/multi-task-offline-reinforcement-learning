import optuna
import d3rlpy
import argparse
import logging
import dill
import numpy as np
from d3rlpy.dataset import MDPDataset
import discrete_SAC_BC

# Specify dataset
dataset_quality = "mixed"
dataset_size = "80"
dataset_path = f"./datasets/{dataset_quality}_{dataset_size}x.pkl"


def objective(trial):


    config = discrete_SAC_BC.Config()
    
    # Define the hyperparameters to tune
    # 500 to 5000 with 500 steps used
    n_steps = trial.suggest_int(name='n_steps', low=40000, high=50000, step=1000)
    learning_rate = trial.suggest_loguniform(name='learning_rate', low=1e-4, high=3e-4)
    beta = trial.suggest_float(name='beta', low=0.8, high=1, step=0.1)

    # Update Config
    config.num_epochs = int(n_steps / config.num_updates_on_epoch)
    config.alpha_learning_rate = learning_rate
    config.actor_learning_rate = learning_rate
    config.critic_learning_rate = learning_rate
    config.beta = beta
    config.eval_every = config.num_epochs * config.num_updates_on_epoch # TODO: maybe no eval needed during training?!
    print("CONFIG:", config)


    performance_list = []
    
    for seed in range(5):
        config.train_seed = seed
        config.eval_seed = seed
        
        # Train Model
        discrete_SAC_BC.train(config=config, dataset_tuple=(dataset_quality, dataset_path), train_seed=seed)

        # Load Model
        model_path = {"sac": f"{config.checkpoints_path}/model_{n_steps}.pt"}

        # Evaluate the model performance
        df_rewards_all_envs = discrete_SAC_BC.eval(config=config, model_paths=model_path, environments=["train"])
        
        # TODO: check that you get only the reward from the last epoch
        rewards_all_envs = df_rewards_all_envs["Reward_mean"]
        performance_train = rewards_all_envs[0]
        performance_list.append(performance_train)
        print("performance_list:", performance_list)
    
    average_performance = np.mean(performance_list)

    return average_performance  # Optuna tries to minimize this value by default



from optuna.visualization import plot_optimization_history, plot_contour, plot_rank, plot_param_importances

def optimize_sac_bc():
    study_name = "tuning_sac_bc"  # Unique identifier of the study.
    storage_name = f"sqlite:///{study_name}_{dataset_quality}.db".format(study_name)
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, load_if_exists=True)
    # study = optuna.create_study(direction='maximize', study_name='sac_bc_tuning')  # Use 'minimize' for loss, 'maximize' for accuracy or other performance metrics
    # study.optimize(objective, n_trials=10)  # Number of trials to perform

    print("Best hyperparameters: ", study.best_trial.params) 
    print("Best value: ", study.best_trial.value)    

    fig = plot_optimization_history(study)
    fig.update_layout(title=f"History: SAC+BC Tuning on {dataset_size}x {dataset_quality.capitalize()} Dataset",)
    fig.write_image(f"results/tuning/{study_name}_{dataset_quality}_history.png") 
    fig = plot_rank(study)
    fig.update_layout(title=f"Rank: SAC+BC Tuning on {dataset_size}x {dataset_quality.capitalize()} Dataset",)
    fig.write_image(f"results/tuning/{study_name}_{dataset_quality}_rank.png")
    fig = plot_contour(study, params=[ "learning_rate", "n_steps", "beta"])
    fig.update_layout(title=f"Contour: SAC+BC Tuning on {dataset_size}x {dataset_quality.capitalize()} Dataset",)
    fig.write_image(f"results/tuning/{study_name}_{dataset_quality}_contour.png") 

    # fig = plot_param_importances(study)
    # fig.update_layout(title=f"Hyperparameter Importance: SAC Tuning on {dataset_size}x {dataset_quality.capitalize()} Dataset",)
    # fig.write_image(f"results/tuning/{study_name}_{dataset_quality}_importance.png") 


    return study.best_trial.params


if __name__ == "__main__":
    optimize_sac_bc()

