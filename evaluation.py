import numpy as np


def eval_policy(policy, env, n_steps=200):
    total_reward = 0

    obs = env.reset()
    for i in range(n_steps):
        action = policy.predict(np.array(obs).reshape(1, -1))

        obs, reward, done, _ = env.step(action)
        if env.spec._env_name == 'MountainCar': 
            # метод потенциалов
            # https://habr.com/ru/company/hsespb/blog/444428/
            # https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
            reward = reward + 200 * (0.4 * abs(obs[1]) - abs(obs[1]))

        total_reward = total_reward + reward

        if done:
            break

    return total_reward