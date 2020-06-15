import gym
import numpy as np
import matplotlib.pyplot as plt

from linear import ThreeLayerNetwork
from evaluation import eval_policy
from es import OpenAiES

from collections import defaultdict
from tqdm import tqdm
from pprint import pprint

# es
LEARNING_RATE = 0.03
NOISE_STD = 0.05
N_SESSIONS = 48
POPULATION_SIZE = 256 # TODO: rename to POPULATION_SIZE
N_ENV_STEPS = 200


def train_loop(policy, env, n_sessions, npop, log_best=False):
    es = OpenAiES(policy, learning_rate=LEARNING_RATE, noise_std=NOISE_STD)

    log = defaultdict(list)
    for session in tqdm(range(n_sessions)):
        population = es.generate_population(npop=npop)

        rewards = np.zeros(npop)
        for i, new_policy in enumerate(population):
            rewards[i] = eval_policy(new_policy, env, n_steps=N_ENV_STEPS)

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
                best_rewards[i] = eval_policy(best_policy, env, n_steps=N_ENV_STEPS)

            log["best_mean_rewards"].append(np.mean(best_rewards))
            log["best_std_rewards"].append(np.std(best_rewards))            

    return log


def run_experiment(config):
    pass


def plot_rewards(mean_rewards, std_rewards, config):
    pass


def main():
    env = gym.make("CartPole-v0")
    
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    policy = ThreeLayerNetwork(
        in_features=n_states, 
        out_features=n_actions, 
        hidden_sizes=(32, 32)
    )
    log = train_loop(policy, env, N_SESSIONS, POPULATION_SIZE, log_best=True)

    best_mean = np.array(log["best_mean_rewards"])
    best_std = np.array(log["best_std_rewards"])

    stats = (
    f"n_sessions: {N_SESSIONS}\npopulation_size: {POPULATION_SIZE}\nlr: {LEARNING_RATE}\nnoise_std: {NOISE_STD}\nenv_steps: {N_ENV_STEPS}"
    )
    
    fig, ax = plt.subplots()
    plt.figure(figsize=(12, 8))
    plt.text(0.35, 1.25, stats, transform=ax.transAxes)
    plt.title("CartPole-v0: Single run")
    plt.plot(np.arange(best_mean.shape[0]), best_mean)
    plt.fill_between(np.arange(best_mean.shape[0]), best_mean + best_std, best_mean - best_std, alpha=0.5)
    plt.xlabel("weights updates (mod 2)")
    plt.ylabel("reward")
    plt.savefig('plots/test_single_v4.png')


if __name__ == "__main__":
    main()