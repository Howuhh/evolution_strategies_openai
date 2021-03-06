import sys
import pickle 

sys.path.append('..')

from training import run_experiment

# solving the task as getting an average score of 300+ over 100 consecutive random trials.
def test():
    test_config = {
        "experiment_name": "test_BipedalWalker_v6.2",
        "plot_path": "../plots/",
        "model_path": "../models/",
        "log_path": "../logs/",
        "init_model": "../models/test_BipedalWalker_v6.1.pkl",
        "env": "BipedalWalker-v3",
        "n_sessions": 250,
        "env_steps": 1300, 
        "population_size": 128,
        "learning_rate": 0.065,
        "noise_std": 0.07783,
        "noise_decay": 0.995,
        "decay_step": 20,
        "eval_step": 10,
        "hidden_sizes": (64, 40) # sizes from https://designrl.github.io/
    }
    
    policy = run_experiment(test_config, n_jobs=4)


if __name__ == "__main__":
    test()