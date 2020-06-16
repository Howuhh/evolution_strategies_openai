import pickle
import sys 

sys.path.append('..')

from training import run_experiment


def test():
    test_config = {
        "experiment_name": "test_single_v6",
        "env": "MountainCar-v0",
        "n_sessions": 256,
        "env_steps": 500, 
        "population_size": 256,
        "learning_rate": 0.05,
        "noise_std": 0.1,
        "hidden_sizes": (16, 16)
    }
    
    policy = run_experiment(test_config)

    with open("../models/test_MountainCar_v6.pkl", "rb") as file:
        pickle.dump(policy, file)


if __name__ == "__main__":
    test()