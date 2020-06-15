import gym
import numpy as np
import matplotlib.pyplot as plt

from linear import ThreeLayerNetwork
from training import train_loop

LEARNING_RATE = 0.01
NOISE_STD = 0.05
N_SESSIONS = 64
N_POPULATIONS = 256
np.random.seed(42)


def test():
    env = gym.make("CartPole-v0")
    
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    run_mean_rewards, run_std_rewards = [], []
    for run in tqdm(range(5)):
        policy = ThreeLayerNetwork(
            in_features=n_states, 
            out_features=n_actions, 
            hidden_sizes=(32, 32)
        )
        mean_rewards, std_rewards = train_loop(policy, env, N_SESSIONS, N_POPULATIONS)

        run_mean_rewards.append(mean_rewards)
        run_std_rewards.append(std_rewards)

    run_mean_rewards = np.array(run_mean_rewards).mean(axis=0)
    run_std_rewards = np.array(run_std_rewards).mean(axis=0)

    stats = (
        f"n_sessions: {N_SESSIONS}\nn_populations: {N_POPULATIONS}\nlr: {LEARNING_RATE}\nnoise_std: {NOISE_STD} "
        )
    
    fig, ax = plt.subplots()
    plt.title("CartPole-v0: Averaged rewards")
    plt.plot(np.arange(N_SESSIONS), run_mean_rewards)
    plt.fill_between(np.arange(N_SESSIONS),
                    run_mean_rewards + run_mean_rewards, 
                    run_mean_rewards - run_mean_rewards, alpha=0.5)
    plt.xlabel("weights updates")
    plt.ylabel("reward")
    plt.text(0.05, 0.8, stats, transform=ax.transAxes)
    plt.savefig('plots/test_averaged_v0.png')


if __name__ == "__main__":
    test()