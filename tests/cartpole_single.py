import sys 

sys.path.append('..')
from training import run_experiment


def test():
    test_config = {
        "experiment_name": "test_cartpole",
        "env": "CartPole-v0",
        "n_sessions": 48,
        "env_steps": 200, 
        "population_size": 256,
        "learning_rate": 0.01,
        "noise_std": 0.05,
        "hidden_sizes": (32, 32)
    }
    policy = run_experiment(test_config)


if __name__ == "__main__":
    test()