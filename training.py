import gym
import pickle
import uuid

import numpy as np

from tqdm import tqdm
from joblib import Parallel
from collections import defaultdict

from gym import wrappers

from linear import ThreeLayerNetwork
from es import OpenAiES
from plot import plot_rewards
from evaluation import eval_policy, eval_policy_delayed

# env: (n_states, n_actions)
ENV_INFO = {
    "CartPole-v0": (4, 2),
    "LunarLander-v2": (8, 4),
    "LunarLanderContinuous-v2": (8, 2),
    "MountainCar-v0": (2, 3),
    "MountainCarContinuous-v0": (2, 1),
    "CarRacing-v0": (96*96*3, 3), # TODO: wrap env to prep pixels & discrete actions
    "BipedalWalker-v3": (24, 4)
}


def train_loop(policy, env, config, n_jobs=1, verbose=True):
    es = OpenAiES(
        model=policy, 
        learning_rate=config["learning_rate"], 
        noise_std=config["noise_std"],
        noise_decay=config.get("noise_decay", 1.0),
        lr_decay=config.get("lr_decay", 1.0),
        decay_step=config.get("decay_step", 50)
    )
    
    log = defaultdict(list)
    for session in tqdm(range(config["n_sessions"])):
        population = es.generate_population(config["population_size"])

        rewards_jobs = (eval_policy_delayed(new_policy, env, config["env_steps"]) for new_policy in population)
        rewards = np.array(Parallel(n_jobs=n_jobs)(rewards_jobs))
        
        es.update_population(rewards)
        
        # best policy stats
        if session % 2 == 0:
            best_policy = es.get_model()

            best_rewards = np.zeros(10)
            for i in range(10):
                best_rewards[i] = eval_policy(best_policy, env, config["env_steps"])

            if verbose:
                print(f"Session: {session}") 
                print(f"Mean reward: {round(np.mean(rewards), 4)}", f"std: {round(np.std(rewards), 3)}")

            log["best_mean_rewards"].append(np.mean(best_rewards))
            log["best_std_rewards"].append(np.std(best_rewards))            

    return log


def run_experiment(config, n_jobs=4, verbose=True):
    env = gym.make(config["env"])
    env._env_name = env.spec._env_name

    n_states, n_actions = ENV_INFO[config["env"]]

    policy = ThreeLayerNetwork(
        in_features=n_states, 
        out_features=n_actions, 
        hidden_sizes=config["hidden_sizes"]
    )
    log = train_loop(policy, env, config, n_jobs, verbose)

    plot_rewards(log["best_mean_rewards"], log["best_std_rewards"], config)

    return policy


def render_policy(model_path, env_name):
    with open(model_path, "rb") as file:
        policy = pickle.load(file)

    model_name = model_path.split("/")[-1].split(".")[0]
    
    for i in range(5):
        env = gym.make(env_name)
        env = wrappers.Monitor(env, f'videos/{model_name}/' + str(uuid.uuid4()), force=True)

        eval_policy(policy, env, n_steps=1000)
        env.close()


if __name__ == "__main__":
    pass
    # render_policy("models/test_CartPole_v1.pkl", "CartPole-v0")
    # render_policy("models/test_LunarLander_v3.pkl", "LunarLander-v2")
    # render_policy("models/test_LunarLanderCont_v1.pkl", "LunarLanderContinuous-v2")
    render_policy("models/test_BipedalWalker_v2.pkl", "BipedalWalker-v3")
    
    # test_config = {
    #     "experiment_name": "test_test",
    #     "plot_path": "plots/",
    #     "model_path": "models/",
    #     "env": "CartPole-v0",
    #     "n_sessions": 128,
    #     "env_steps": 200, 
    #     "population_size": 256,
    #     "learning_rate": 0.01,
    #     "noise_std": 0.075,
    #     "hidden_sizes": (64, 64)
    # }

