import sys 
import pickle

sys.path.append('..')
from training import run_experiment


# TODO: parallel & continuous
def test():
    test_config = {
        "experiment_name": "test_LunarLanderCont_v1",
        "plot_path": "../plots/",
        "model_path": "../models/",
        "env": "LunarLanderContinuous-v2",
        "n_sessions": 512,
        "env_steps": 500, 
        "population_size": 256,
        "learning_rate": 0.01,
        "noise_std": 0.075,
        "hidden_sizes": (64, 64)
    }
    policy = run_experiment(test_config)

    with open(f"{test_config['model_path']}{test_config['experiment_name']}.pkl", "wb") as file:
        pickle.dump(policy, file)


if __name__ == "__main__":
    test()
