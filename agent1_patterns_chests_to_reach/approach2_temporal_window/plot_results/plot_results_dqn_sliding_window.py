import matplotlib.pyplot as plt

def plot_learning_curves_dqn(
    training_loss,
    eval_rewards=None,
    success_rates=None,
    eval_interval=10,
    gamma=None,
    window_size=None,
    save_path=None
):
    """
    Plot the DQN training loss, evaluation rewards, and success rates.

    Args:
        training_loss (list): List of training loss values (one per episode).
        eval_rewards (list, optional): Average evaluation rewards recorded 
            every `eval_interval` episodes.
        success_rates (list, optional): Success rates (%) recorded 
            every `eval_interval` episodes.
        eval_interval (int): Interval (in episodes) at which evaluation metrics 
            are computed. Used to plot x-axis for eval metrics.
        gamma (float, optional): Discount factor (used in title, if provided).
        window_size (int, optional): If using a sliding window for states, 
            specify the number of events. This is purely for annotation 
            in the plot title.
        save_path (str, optional): If given, saves the figure at this path; 
            otherwise calls plt.show().
    """
    # Determine how many subplots we need
    num_plots = 1 + int(eval_rewards is not None) + int(success_rates is not None)
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

    # Convert axs to a list if there's only one subplot
    if num_plots == 1:
        axs = [axs]

    # --- Build the main title for the first plot
    title_str = "DQN Training Loss"
    if gamma is not None:
        title_str += f" (gamma={gamma})"
    if window_size is not None:
        title_str += f", window={window_size}"

    # 1) Plot training loss
    axs[0].plot(training_loss, label="Training Loss")
    axs[0].set_title(title_str)
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    current_plot_index = 1

    # 2) Plot evaluation rewards if provided
    if eval_rewards is not None:
        eval_x = range(0, len(eval_rewards) * eval_interval, eval_interval)
        axs[current_plot_index].plot(eval_x, eval_rewards, label="Evaluation Reward", color='orange')
        axs[current_plot_index].set_title("Evaluation Rewards")
        axs[current_plot_index].set_xlabel("Episode")
        axs[current_plot_index].set_ylabel("Avg Reward")
        axs[current_plot_index].legend()
        axs[current_plot_index].grid(True)
        current_plot_index += 1

    # 3) Plot success rates if provided
    if success_rates is not None:
        eval_x = range(0, len(success_rates) * eval_interval, eval_interval)
        axs[current_plot_index].plot(eval_x, success_rates, label="Success Rate (%)", color='purple')
        axs[current_plot_index].set_title("Success Rate over Time")
        axs[current_plot_index].set_xlabel("Episode")
        axs[current_plot_index].set_ylabel("Success %")
        axs[current_plot_index].legend()
        axs[current_plot_index].grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path)
        print(f"Saved DQN plot to {save_path}")
    else:
        plt.show()
