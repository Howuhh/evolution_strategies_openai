import numpy as np

from joblib import delayed


def eval_policy(policy, env, n_steps=200):
    total_reward = 0

    obs = env.reset()
    for i in range(n_steps):
        if env.spec._env_name in ('LunarLanderContinuous', "MountainCarContinuous"):
            action = policy.predict(np.array(obs).reshape(1, -1), scale="tanh")
        else:
            action = policy.predict(np.array(obs).reshape(1, -1), scale="softmax")

        new_obs, reward, done, _ = env.step(action)
        
        # if env.spec._env_name == 'MountainCarContinuous': 
            # метод потенциалов https://habr.com/ru/company/hsespb/blog/444428/
            # reward = reward + 300 * (0.99 * abs(new_obs[1]) - abs(obs[1]))

        total_reward = total_reward + reward
        obs = new_obs

        if done:
            break

    return total_reward


# for parallel
eval_policy_delayed = delayed(eval_policy)