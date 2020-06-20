import numpy as np

from joblib import delayed

CONTINUOUS_ENVS = ('LunarLanderContinuous', "MountainCarContinuous", "BipedalWalker")

def eval_policy(policy, env, n_steps=200):
    total_reward = 0
    print(env)
    
    obs = env.reset()
    for i in range(n_steps):
        if env.spec._env_name in CONTINUOUS_ENVS:
            action = policy.predict(np.array(obs).reshape(1, -1), scale="tanh")
        else:
            action = policy.predict(np.array(obs).reshape(1, -1), scale="softmax")

        new_obs, reward, done, _ = env.step(action)
        
        # if env.spec._env_name == 'MountainCarContinuous':
        #     reward = reward + 10 * abs(new_obs[1])
            # TODO: add novelity search reward (https://lilianweng.github.io/lil-log/2019/09/05/evolution-strategies.html)
            # метод потенциалов https://habr.com/ru/company/hsespb/blog/444428/
            # reward = reward + 300 * (0.99 * abs(new_obs[1]) - abs(obs[1]))

        total_reward = total_reward + reward
        obs = new_obs

        if done:
            break

    return total_reward


# for parallel
eval_policy_delayed = delayed(eval_policy)