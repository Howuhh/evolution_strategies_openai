import numpy as np


def eval_policy(policy, env, n_steps=200):
    total_reward = 0

    obs = env.reset()

    for i in range(n_steps):
        action = policy.predict(np.array(obs).reshape(1, -1))

        obs, reward, done, _ = env.step(action)
        total_reward = total_reward + reward

        if done:
            break

    return total_reward