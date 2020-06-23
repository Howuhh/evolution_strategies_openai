import numpy as np
import matplotlib.pyplot as plt


def plot_rewards(mean_rewards, std_rewards, config):
    best_mean = np.array(mean_rewards)
    best_std = np.array(std_rewards)

    stats = (
    f"""
    n_sessions: {config["n_sessions"]}
    population_size: {config["population_size"]}
    lr: {config["learning_rate"]}
    noise_std: {config["noise_std"]}
    env_steps: {config["env_steps"]}
    """
    ) # TODO: add hidden size info on plot 
    
    fig, ax = plt.subplots()
    plt.figure(figsize=(12, 8))
    plt.text(0.35, 1.25, stats, transform=ax.transAxes)
    plt.title(f"{config['env']}: {config['experiment_name']}") 
    plt.plot(np.arange(best_mean.shape[0]), best_mean)
    plt.fill_between(np.arange(best_mean.shape[0]), best_mean + best_std, best_mean - best_std, alpha=0.5)
    plt.xlabel(f"weights updates (mod {config.get('eval_step', '2')})")
    plt.ylabel("reward")
    plt.savefig(f"{config['plot_path']}{config['experiment_name']}.png")