import gym

import numpy as np
import matplotlib.pyplot as plt

from linear import ThreeLayerNetwork
from evaluation import eval_policy
from es import OpenAiES

from collections import defaultdict
from tqdm import tqdm


def train_loop(policy, env, config, log_best=True):
    es = OpenAiES(policy, learning_rate=config["learning_rate"], noise_std=config["noise_std"])

    log = defaultdict(list)
    for session in tqdm(range(config["n_sessions"])):
        population = es.generate_population(npop=config["population_size"])

        rewards = np.zeros(config["population_size"])
        for i, new_policy in enumerate(population):
            rewards[i] = eval_policy(new_policy, env, n_steps=config["env_steps"])

        es.update_population(rewards)

        print(f"Session: {session}") 
        print(np.mean(rewards))
        print(np.std(rewards))
        
        # training stats
        log["pop_mean_rewards"].append(np.mean(rewards))
        log["pop_std_rewards"].append(np.std(rewards))

        # best policy stats
        if log_best and session % 2 == 0:
            best_policy = es.get_model()

            best_rewards = np.zeros(10)
            for i in range(10):
                best_rewards[i] = eval_policy(best_policy, env, n_steps=config["env_steps"])

            log["best_mean_rewards"].append(np.mean(best_rewards))
            log["best_std_rewards"].append(np.std(best_rewards))            

    return log


def run_experiment(config):
    env = gym.make("CartPole-v0")

    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    policy = ThreeLayerNetwork(
        in_features=n_states, 
        out_features=n_actions, 
        hidden_sizes=config["hidden_sizes"]
    )
    log = train_loop(policy, env, config)

    plot_rewards(log["best_mean_rewards"], log["best_std_rewards"], config)


def plot_rewards(mean_rewards, std_rewards, config):
    best_mean = np.array(mean_rewards)
    best_std = np.array(std_rewards)

    stats = (
    f"""
    n_sessions: {config["n_sessions"]}
    population_size: {config["population_size"]}
    lr: {config["learning_rate"]}
    noise_std: {config["noise_std"]}
    env_steps: {config["env_steps"]}
    """
    )
    
    fig, ax = plt.subplots()
    plt.figure(figsize=(12, 8))
    plt.text(0.35, 1.25, stats, transform=ax.transAxes)
    plt.title("CartPole-v0: Single run")
    plt.plot(np.arange(best_mean.shape[0]), best_mean)
    plt.fill_between(np.arange(best_mean.shape[0]), best_mean + best_std, best_mean - best_std, alpha=0.5)
    plt.xlabel("weights updates (mod 2)")
    plt.ylabel("reward")
    plt.savefig(f'plots/{config["experiment_name"]}.png')


def main():
    test_config = {
        "experiment_name": "test_single_v5",
        "n_sessions": 48,
        "env_steps": 200, 
        "population_size": 256,
        "learning_rate": 0.03,
        "noise_std": 0.05,
        "hidden_sizes": (32, 32)
    }
    run_experiment(test_config)


if __name__ == "__main__":
    main()