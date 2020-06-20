import sys
import pickle 

sys.path.append('..')

from training import run_experiment

# ISSUE: deepcopy(env) not copy .spec and .spec._env_name, so wait
def test():
    test_config = {
        "experiment_name": "test_BipedalWalker_v1",
        "plot_path": "../plots/",
        "model_path": "../models/",
        "env": "BipedalWalker-v2",
        "n_sessions": 5,
        "env_steps": 1600, 
        "population_size": 256,
        "learning_rate": 0.05,
        "noise_std": 0.05,
        "hidden_sizes": (64, 64)
    }
    
    policy = run_experiment(test_config, n_jobs=2)

    with open(f"{test_config['model_path']}{test_config['experiment_name']}.pkl", "wb") as file:
        pickle.dump(policy, file)


if __name__ == "__main__":
    test()