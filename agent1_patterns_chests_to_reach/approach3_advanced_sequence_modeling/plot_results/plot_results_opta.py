# agent1_patterns_chests_to_reach/approach3_advanced_sequence_modeling/plot_results/plot_results_opta.py
"""
Helper to visualize training of an OPTA (On-Policy Transformer Actor) agent.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_learning_curves_opta(
    policy_losses : List[float],
    entropies     : Optional[List[float]] = None,
    eval_rewards  : Optional[List[float]] = None,
    success_rates : Optional[List[float]] = None,
    eval_interval : int   = 100,
    lr            : float | None = None,
    d_model       : int   | None = None,
    n_layers      : int   | None = None,
    n_heads       : int   | None = None,
    save_path     : str   | None = None,
):
    """
    Parameters
    ----------
    policy_losses : list of REINFORCE losses (one entry per update)
    entropies     : policy entropy (optional)
    eval_rewards  : average eval rewards (optional)
    success_rates : eval success rates in % (optional)
    eval_interval : number of episodes between evaluations
    lr            : learning rate (display only)
    d_model       : transformer width (optional)
    n_layers      : number of transformer layers (optional)
    n_heads       : number of attention heads (optional)
    save_path     : file path to save the figure (else will display)
    """
    extra_eval_plots = int(eval_rewards is not None) + int(success_rates is not None)
    extra_entropy    = int(entropies is not None)
    n_plots = 1 + extra_entropy + extra_eval_plots

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    # 1) Policy loss
    title = "OPTA – policy loss"
    if lr is not None:
        title += f" (lr={lr})"
    if d_model is not None:
        title += f" – d_model={d_model}"
    if n_layers is not None:
        title += f" – layers={n_layers}"
    if n_heads is not None:
        title += f" – heads={n_heads}"

    axes[0].plot(policy_losses, label="Policy loss")
    axes[0].set_title(title)
    axes[0].set_xlabel("Update step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    idx = 1

    # 2) Entropy
    if entropies is not None:
        axes[idx].plot(entropies, label="Entropy", color="tab:green")
        axes[idx].set_title("Policy entropy")
        axes[idx].set_xlabel("Update step")
        axes[idx].set_ylabel("Entropy")
        axes[idx].grid(True)
        axes[idx].legend()
        idx += 1

    # 3) Eval reward
    if eval_rewards is not None:
        x = range(0, len(eval_rewards) * eval_interval, eval_interval)
        axes[idx].plot(x, eval_rewards, label="Avg eval reward", color="tab:red")
        axes[idx].set_title("Evaluation – average reward")
        axes[idx].set_xlabel("Episode")
        axes[idx].set_ylabel("Reward")
        axes[idx].grid(True)
        axes[idx].legend()
        idx += 1

    # 4) Success %
    if success_rates is not None:
        x = range(0, len(success_rates) * eval_interval, eval_interval)
        axes[idx].plot(x, success_rates, label="Success rate %", color="tab:purple")
        axes[idx].set_title("Evaluation – success rate")
        axes[idx].set_xlabel("Episode")
        axes[idx].set_ylabel("Success %")
        axes[idx].grid(True)
        axes[idx].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"[plot] Figure saved to {save_path}")
    else:
        plt.show()
