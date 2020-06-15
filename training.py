import gym
import numpy as np
import matplotlib.pyplot as plt

from linear import ThreeLayerNetwork
from evaluation import eval_policy
from es import OpenAiES
from tqdm import tqdm

# es
LEARNING_RATE = 0.01
NOISE_STD = 0.05
N_SESSIONS = 64
N_POPULATIONS = 256
np.random.seed(42)

def train_loop(policy, env, n_sessions, npop):
    es = OpenAiES(policy, learning_rate=LEARNING_RATE, noise_std=NOISE_STD)

    mean_rewards, std_rewards = [], []
    for session in tqdm(range(n_sessions)):
        population = es.generate_population(npop=npop)

        rewards = np.zeros(npop)
        for i, new_policy in enumerate(population):
            rewards[i] = eval_policy(new_policy, env, n_iter=500)

        es.update_population(rewards)     
        
        # training stats
        mean_rewards.append(np.mean(rewards))
        std_rewards.append(np.std(rewards))

    return np.array(mean_rewards), np.array(std_rewards)


def main():
    env = gym.make("CartPole-v0")
    
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    policy = ThreeLayerNetwork(
        in_features=n_states, 
        out_features=n_actions, 
        hidden_sizes=(32, 32)
    )
    mean_rewards, std_rewards = train_loop(policy, env, N_SESSIONS, N_POPULATIONS)

    stats = (
    f"n_sessions: {N_SESSIONS}\nn_populations: {N_POPULATIONS}\nlr: {LEARNING_RATE}\nnoise_std: {NOISE_STD} "
    )
    
    fig, ax = plt.subplots()
    plt.title("CartPole-v0: Single run")
    plt.plot(np.arange(N_SESSIONS), mean_rewards)
    plt.fill_between(np.arange(N_SESSIONS),
                    mean_rewards + std_rewards, 
                    mean_rewards - std_rewards, alpha=0.5)
    plt.xlabel("weights updates")
    plt.ylabel("reward")
    plt.text(0.05, 0.8, stats, transform=ax.transAxes)
    plt.savefig('plots/run_single_v0.png')

    # run_mean_rewards, run_std_rewards = [], []
    # for run in tqdm(range(5)):
    #     policy = ThreeLayerNetwork(
    #         in_features=n_states, 
    #         out_features=n_actions, 
    #         hidden_sizes=(32, 32)
    #     )
    #     mean_rewards, std_rewards = train_loop(policy, env, N_SESSIONS, N_POPULATIONS)

    #     run_mean_rewards.append(mean_rewards)
    #     run_std_rewards.append(std_rewards)

    # run_mean_rewards = np.array(run_mean_rewards).mean(axis=0)
    # run_std_rewards = np.array(run_std_rewards).mean(axis=0)

    # stats = (
    #     f"n_sessions: {N_SESSIONS}\nn_populations: {N_POPULATIONS}\nlr: {LEARNING_RATE}\nnoise_std: {NOISE_STD} "
    #     )
    
    # fig, ax = plt.subplots()
    # plt.title("CartPole-v0: Averaged rewards")
    # plt.plot(np.arange(N_SESSIONS), run_mean_rewards)
    # plt.fill_between(np.arange(N_SESSIONS),
    #                 run_mean_rewards + run_mean_rewards, 
    #                 run_mean_rewards - run_mean_rewards, alpha=0.5)
    # plt.xlabel("weights updates")
    # plt.ylabel("reward")
    # plt.text(0.05, 0.8, stats, transform=ax.transAxes)
    # plt.savefig('plots/run_v0.png')


if __name__ == "__main__":
    main()