import sys
import pickle 

sys.path.append('..')

from training import run_experiment

# solving the task as getting an average score of 300+ over 100 consecutive random trials.
def test():
    test_config = {
        "experiment_name": "test_BipedalWalker_v2",
        "plot_path": "../plots/",
        "model_path": "../models/",
        "env": "BipedalWalker-v3",
        "n_sessions": 256,
        "env_steps": 1600, 
        "population_size": 256,
        "learning_rate": 0.065,
        "noise_std": 0.08,
        "hidden_sizes": (64, 64)
    }
    
    policy = run_experiment(test_config, n_jobs=4)

    with open(f"{test_config['model_path']}{test_config['experiment_name']}.pkl", "wb") as file:
        pickle.dump(policy, file)


if __name__ == "__main__":
    test()