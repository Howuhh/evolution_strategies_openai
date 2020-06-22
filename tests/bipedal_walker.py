import sys
import pickle 

sys.path.append('..')

from training import run_experiment

# solving the task as getting an average score of 300+ over 100 consecutive random trials.
def test():
    test_config = {
        "experiment_name": "test_BipedalWalker_v4",
        "plot_path": "../plots/",
        "model_path": "../models/",
        "env": "BipedalWalker-v3",
        "n_sessions": 512,
        "env_steps": 1000, 
        "population_size": 128,
        "learning_rate": 0.03,
        "noise_std": 0.2,
        "noise_decay": 0.95,
        "decay_step": 20,
        "eval_steps": 10,
        "hidden_sizes": (40, 40) # sizes from https://designrl.github.io/
    }
    
    policy = run_experiment(test_config, n_jobs=4)

    with open(f"{test_config['model_path']}{test_config['experiment_name']}.pkl", "wb") as file:
        pickle.dump(policy, file)


if __name__ == "__main__":
    test()