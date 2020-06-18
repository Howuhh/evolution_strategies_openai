import gym
import numpy as np

from linear import ThreeLayerNetwork
from evaluation import eval_policy
from es import OpenAiES
from plot import plot_rewards

from collections import defaultdict
from tqdm import tqdm


def train_loop(policy, env, config):
    es = OpenAiES(policy, learning_rate=config["learning_rate"], noise_std=config["noise_std"])

    log = defaultdict(list)
    for session in tqdm(range(config["n_sessions"])):
        population = es.generate_population(npop=config["population_size"])

        rewards = np.zeros(config["population_size"])
        for i, new_policy in enumerate(population):
            rewards[i] = eval_policy(new_policy, env, n_steps=config["env_steps"])

        es.update_population(rewards)
        
        # best policy stats
        if session % 2 == 0:
            best_policy = es.get_model()

            best_rewards = np.zeros(10)
            for i in range(10):
                best_rewards[i] = eval_policy(best_policy, env, n_steps=config["env_steps"])

            print(f"Session: {session}") 
            print(f"Mean reward: {round(np.mean(rewards), 4)}", f"std: {round(np.std(rewards), 3)}")

            log["best_mean_rewards"].append(np.mean(best_rewards))
            log["best_std_rewards"].append(np.std(best_rewards))            

    return log


def run_experiment(config):
    env = gym.make(config["env"])

    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    policy = ThreeLayerNetwork(
        in_features=n_states, 
        out_features=n_actions, 
        hidden_sizes=config["hidden_sizes"]
    )
    log = train_loop(policy, env, config)

    plot_rewards(log["best_mean_rewards"], log["best_std_rewards"], config)

    return policy
