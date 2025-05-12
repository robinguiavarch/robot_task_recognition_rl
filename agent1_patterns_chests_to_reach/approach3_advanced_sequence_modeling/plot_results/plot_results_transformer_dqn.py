# plot_results_transformer_dqn.py

import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_results_transformer_dqn(
    training_loss,
    eval_rewards=None,
    success_rates=None,
    eval_interval=50,
    gamma=None,
    window_size=None,
    save_path=None
):
    n_plots = 1 + int(eval_rewards is not None) + int(success_rates is not None)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    # Loss plot
    title = "Transformer–DQN Training Loss"
    if gamma is not None:
        title += f" (γ={gamma})"
    if window_size is not None:
        title += f", window={window_size}"
    axes[0].plot(training_loss, label="Loss")
    axes[0].set_title(title)
    axes[0].set_xlabel("Update step")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True)

    idx = 1
    # Eval rewards
    if eval_rewards is not None:
        x = np.arange(len(eval_rewards)) * eval_interval
        axes[idx].plot(x, eval_rewards, color="C1", label="Eval reward")
        axes[idx].set_title("Evaluation Rewards")
        axes[idx].set_xlabel("Episode")
        axes[idx].set_ylabel("Avg Reward")
        axes[idx].legend()
        axes[idx].grid(True)
        idx += 1

    # Success rates
    if success_rates is not None:
        x = np.arange(len(success_rates)) * eval_interval
        axes[idx].plot(x, success_rates, color="C2", label="Success %")
        axes[idx].set_title("Success Rate Over Time")
        axes[idx].set_xlabel("Episode")
        axes[idx].set_ylabel("Success %")
        axes[idx].legend()
        axes[idx].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()



