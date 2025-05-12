# agent1_patterns_chests_to_reach/approach2_temporal_window/plot_results/plot_results_actor_lstm.py
"""
Helper to visualise training of a REINFORCE-LSTM (policy-only) agent.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_learning_curves_actor_lstm(
    policy_losses : List[float],
    entropies     : Optional[List[float]] = None,
    eval_rewards  : Optional[List[float]] = None,
    success_rates : Optional[List[float]] = None,
    eval_interval : int   = 100,
    lr            : float | None = None,
    lstm_hidden   : int   | None = None,
    save_path     : str   | None = None,
):
    """
    Parameters
    ----------
    policy_losses : liste de la loss REINFORCE (une entrée par mise-à-jour)
    entropies     : entropie de la policy (optionnel)
    eval_rewards  : récompenses moyennes d’éval (optionnel)
    success_rates : taux de succès d’éval en % (optionnel)
    eval_interval : espacement (en nb d’épisodes) entre deux évaluations
    lr            : learning-rate, affiché pour info
    lstm_hidden   : taille du hidden LSTM
    save_path     : chemin pour sauvegarder la figure ; sinon plt.show()
    """
    extra_eval_plots  = int(eval_rewards is not None) + int(success_rates is not None)
    extra_entropy     = int(entropies is not None)
    n_plots = 1 + extra_entropy + extra_eval_plots

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    # 1) policy loss
    title = "Actor-only : policy loss"
    if lr is not None:
        title += f" (lr={lr})"
    if lstm_hidden is not None:
        title += f"  – LSTM hidden={lstm_hidden}"

    axes[0].plot(policy_losses, label="Policy loss")
    axes[0].set_title(title)
    axes[0].set_xlabel("Update step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    idx = 1

    # 2) entropy
    if entropies is not None:
        axes[idx].plot(entropies, label="Entropy", color="tab:green")
        axes[idx].set_title("Policy entropy")
        axes[idx].set_xlabel("Update step")
        axes[idx].set_ylabel("Entropy")
        axes[idx].grid(True)
        axes[idx].legend()
        idx += 1

    # 3) eval reward
    if eval_rewards is not None:
        x = range(0, len(eval_rewards) * eval_interval, eval_interval)
        axes[idx].plot(x, eval_rewards, label="Avg eval reward", color="tab:red")
        axes[idx].set_title("Récompense moyenne (évaluation)")
        axes[idx].set_xlabel("Episode")
        axes[idx].set_ylabel("Reward")
        axes[idx].grid(True)
        axes[idx].legend()
        idx += 1

    # 4) success %
    if success_rates is not None:
        x = range(0, len(success_rates) * eval_interval, eval_interval)
        axes[idx].plot(x, success_rates, label="Success rate %", color="tab:purple")
        axes[idx].set_title("Taux de succès (évaluation)")
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
