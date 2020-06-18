import sys 
import pickle

sys.path.append('..')
from training import run_experiment


def test():
    test_config = {
        "experiment_name": "test_CartPole_v2",
        "plot_path": "../plots/",
        "model_path": "../models/",
        "env": "CartPole-v0",
        "n_sessions": 64,
        "env_steps": 200, 
        "population_size": 256,
        "learning_rate": 0.01,
        "noise_std": 0.05,
        "hidden_sizes": (64, 64)
    }
    policy = run_experiment(test_config)

    # TODO: not easy, need a change
    with open(f"{test_config['model_path']}{test_config['experiment_name']}.pkl", "wb") as file:
        pickle.dump(policy, file)


if __name__ == "__main__":
    test()