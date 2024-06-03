import optuna
import d3rlpy
import argparse
import logging
import dill
import numpy as np
from d3rlpy.dataset import MDPDataset
import discrete_SAC_N

def objective(trial):
    dataset_path = "./datasets/dataset_gen_optimal_policy_40x.pkl"
    dataset_quality = "optimal"
    
    config = discrete_SAC_N.Config()
    
    # Define the hyperparameters to tune
    n_steps = trial.suggest_int('n_steps', 5000, 30000, 2500)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 3e-4)
    gamma=trial.suggest_uniform('gamma', 0.98, 0.99)

    # Update Config
    config.num_epochs = int(n_steps / config.num_updates_on_epoch)
    config.alpha_learning_rate = learning_rate
    config.actor_learning_rate = learning_rate
    config.critic_learning_rate = learning_rate
    config.gamma = gamma
    config.eval_every = config.num_epochs # TODO: maybe no eval needed during training?!
    print("CONFIG:", config)

    # Train Model
    discrete_SAC_N.train(config=config, dataset_tuple=(dataset_quality, dataset_path))

    # Load Model
    model_path = {"sac": f"{config.checkpoints_path}/{dataset_quality}_{n_steps}.pt"}

    # Evaluate the model performance
    df_rewards = discrete_SAC_N.eval(config=config, model_paths=model_path)
    
    # TODO: check that you get only the reward from the last epoch
    rewards = df_rewards["Reward_mean"]
    print("REWARDS:", rewards)
    performance = np.mean(rewards)

    return performance  # Optuna tries to minimize this value by default



from optuna.visualization import plot_optimization_history, plot_contour, plot_rank

def optimize_sac():
    study_name = "tuning_sac"  # Unique identifier of the study.
    storage_name = "sqlite:///tuning_sac.db".format(study_name)
    study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, load_if_exists=True)
    # study = optuna.create_study(direction='maximize', study_name='sac_tuning')  # Use 'minimize' for loss, 'maximize' for accuracy or other performance metrics
    # study.optimize(objective, n_trials=20)  # Number of trials to perform

    print("Best hyperparameters: ", study.best_trial.params) 
    print("Best value: ", study.best_trial.value)    

    fig = plot_optimization_history(study)
    fig.update_layout(title="History: SAC Tuning on 40x Optimal (100%) Dataset",)
    fig.write_image(f"results/{study_name}_history.png") 
    fig = plot_rank(study)
    fig.update_layout(title="Rank: SAC Tuning on 40x Optimal (100%) Dataset",)
    fig.write_image(f"results/{study_name}_rank.png")
    fig = plot_contour(study, params=[ "learning_rate", "n_steps", "gamma"])
    fig.update_layout(title="Contour: SAC Tuning on 40x Optimal (100%) Dataset",)
    fig.write_image(f"results/{study_name}_contour.png") 

    return study.best_trial.params


if __name__ == "__main__":
    optimize_sac()

