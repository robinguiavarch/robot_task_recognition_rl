import matplotlib.pyplot as plt

def plot_learning_curves(actor_losses, critic_losses, eval_rewards=None, success_rates=None, eval_interval=10, gamma=None, save_path=None):
    """
    Plot actor and critic losses, evaluation rewards, and success rates.

    Args:
        actor_losses (list): List of actor loss values per episode.
        critic_losses (list): List of critic loss values per episode.
        eval_rewards (list, optional): Evaluation rewards at every `eval_interval`.
        success_rates (list, optional): Success rates at every `eval_interval`.
        eval_interval (int): Interval used for evaluation.
        gamma (float, optional): Discount factor (used in title).
        save_path (str, optional): If provided, saves the plot to this path.
    """
    num_plots = 1 + int(eval_rewards is not None) + int(success_rates is not None)
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

    axs[0].plot(actor_losses, label="Actor Loss", color='blue')
    axs[0].plot(critic_losses, label="Critic Loss", color='green')
    axs[0].set_title(f"Training Losses{' (gamma=' + str(gamma) + ')' if gamma else ''}")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    plot_index = 1

    if eval_rewards is not None:
        eval_x = list(range(0, len(eval_rewards) * eval_interval, eval_interval))
        axs[plot_index].plot(eval_x, eval_rewards, label="Evaluation Reward", color='orange')
        axs[plot_index].set_title("Evaluation Rewards")
        axs[plot_index].set_xlabel("Episode")
        axs[plot_index].set_ylabel("Avg Reward")
        axs[plot_index].legend()
        axs[plot_index].grid(True)
        plot_index += 1

    if success_rates is not None:
        eval_x = list(range(0, len(success_rates) * eval_interval, eval_interval))
        axs[plot_index].plot(eval_x, success_rates, label="Success Rate (%)", color='purple')
        axs[plot_index].set_title("Success Rate over Time")
        axs[plot_index].set_xlabel("Episode")
        axs[plot_index].set_ylabel("Success %")
        axs[plot_index].legend()
        axs[plot_index].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
