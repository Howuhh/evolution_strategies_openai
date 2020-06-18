import gym
import pickle
import uuid

import numpy as np

from gym import wrappers

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
        # TODO: parallel (?)
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


# TODO: add code to record video/render of agent in env
def render_policy(model_path, env_name):
    with open(model_path, "rb") as file:
        policy = pickle.load(file)

    model_name = model_path.split("/")[-1].split(".")[0]
    
    for i in range(10):
        env = gym.make(env_name)
        env = wrappers.Monitor(env, f'videos/{model_name}/' + str(uuid.uuid4()), force=True)

        eval_policy(policy, env, n_steps=800)
    
        env.close()


if __name__ == "__main__":
    # render_policy("models/test_LunarLander_v2.pkl", "LunarLander-v2")
    render_policy("models/test_CartPole_v1.pkl", "CartPole-v0")
    # render_policy("models/test_LunarLander_v3.pkl", "LunarLander-v2")
