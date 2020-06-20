import sys
import pickle 

sys.path.append('..')

from training import run_experiment

# MountainCar-v0 defines "solving" as getting average reward of -110.0 over 100 consecutive trials.
# TODO: wait for novelity search
def test():
    test_config = {
        "experiment_name": "test_MountainCarCont_v2",
        "plot_path": "../plots/",
        "model_path": "../models/",
        "env": "MountainCarContinuous-v0",
        "n_sessions": 128,
        "env_steps": 200, 
        "population_size": 256,
        "learning_rate": 0.1,
        "noise_std": 0.5,
        "hidden_sizes": (32, 32)
    }
    
    policy = run_experiment(test_config, n_jobs=4)

    with open(f"{test_config['model_path']}{test_config['experiment_name']}.pkl", "wb") as file:
        pickle.dump(policy, file)


if __name__ == "__main__":
    test()
