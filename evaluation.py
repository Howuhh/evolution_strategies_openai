import numpy as np

from joblib import delayed

CONTINUOUS_ENVS = ('LunarLanderContinuous', "MountainCarContinuous", "BipedalWalker")

def eval_policy(policy, env, n_steps=200):
    try:
        env_name = env.spec._env_name
    except AttributeError:
        env_name = env._env_name
    
    total_reward = 0
    
    obs = env.reset()
    for i in range(n_steps):
        if env_name in CONTINUOUS_ENVS:
            action = policy.predict(np.array(obs).reshape(1, -1), scale="tanh")
        else:
            action = policy.predict(np.array(obs).reshape(1, -1), scale="softmax")

        new_obs, reward, done, _ = env.step(action)
        
        total_reward = total_reward + reward
        obs = new_obs

        if done:
            break

    return total_reward


# for parallel
eval_policy_delayed = delayed(eval_policy)