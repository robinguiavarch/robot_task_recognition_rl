# train_agent_dqn.py

import gym
import torch
from torch.optim import Adam
import numpy as np
from pathlib import Path
from agent1_patterns_chests_to_reach.env.register_envs import (
    register_custom_envs,
    all_types,
    all_attributes,
)

# Import the sliding-window tools
from agent1_patterns_chests_to_reach.approach2_temporal_window.encoder.encoders_sliding_window import (
    sliding_window_encoder,
    action_encoder,
    convert_index_to_action,
    reset_sliding_window,
)
from agent1_patterns_chests_to_reach.utils.event_encoding import (
    event_to_dict_from_gym as event_from_obs_gym,
)
from agent1_patterns_chests_to_reach.approach2_temporal_window.agents.replay_buffer_sliding_window import ReplayBuffer
from agent1_patterns_chests_to_reach.approach2_temporal_window.evaluation.evaluation_sliding_window import evaluate_dqn
from agent1_patterns_chests_to_reach.approach2_temporal_window.agents.dqn_agent_sliding_window import (
    QNetwork,
    select_action,
    train,
)

def train_agent(
    env_name="OpenTheChests-v0",
    buffer_capacity=1000,
    epochs=500,
    batch_size=128,
    learning_rate=0.01,
    gamma=0.99,
    evaluation_interval=50,
    evaluation_episodes=100,
    eps=0.1,
):
    """
    Trains a DQN agent on the specified environment (e.g. 'OpenTheChests-v0'),
    using a sliding-window encoder for states.

    In this approach:
      - Each single event is encoded into a 33D vector (symbol + bg + fg + start/end time).
      - We keep a window of up to 4 events, resulting in a 132D state (4 x 33).
      - The action space has 8 possible actions (3-bit, i.e. 2^3 = 8).

    Args:
        env_name (str): Name of the Gym environment (e.g., 'OpenTheChests-v0').
        buffer_capacity (int): Max capacity of the replay buffer.
        epochs (int): Number of training episodes.
        batch_size (int): Batch size for training steps.
        learning_rate (float): Learning rate for Adam optimizer.
        gamma (float): Discount factor.
        evaluation_interval (int): Evaluate the agent every `evaluation_interval` episodes.
        evaluation_episodes (int): Number of episodes for each evaluation.
        eps (float): Epsilon for epsilon-greedy exploration.

    Returns:
        (training_loss, average_rewards, average_successes):
            - training_loss (list[float]): Loss values per training step.
            - average_rewards (list[float]): Average reward at each evaluation.
            - average_successes (list[float]): Success rate (%) at each evaluation.
    """

    # 1) Register custom environments
    register_custom_envs()

    # 2) Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) State is 132D: 4 events × 33D each
    state_size = 33 * 4  
    action_size = 8      # 3-bit actions => 2^3 = 8

    # 4) Create environment and networks
    env = gym.make(env_name)
    q_net = QNetwork(state_size, action_size).to(device)
    target_q_net = QNetwork(state_size, action_size).to(device)
    target_q_net.load_state_dict(q_net.state_dict())

    # 5) Optimizer and replay buffer
    optimizer = Adam(q_net.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(buffer_capacity)

    # 6) Initial reset + clearing the sliding window
    obs = env.reset()
    reset_sliding_window()

    # Convert first observation -> dict -> 33D -> sliding window -> 132D
    event_dict = event_from_obs_gym(obs, all_types, all_attributes)
    state = sliding_window_encoder(event_dict, all_types, all_attributes)
    done = False

    # Lists to store training metrics
    training_loss = []
    average_rewards = []
    average_successes = []

    for episode in range(epochs):
        if done:
            # Reset env & sliding window for a new episode
            obs = env.reset()
            reset_sliding_window()

            event_dict = event_from_obs_gym(obs, all_types, all_attributes)
            state = sliding_window_encoder(event_dict, all_types, all_attributes)
            done = False

        # Epsilon-greedy action selection
        action_index = select_action(q_net, state, eps=eps, device=device)
        action_bits = convert_index_to_action(action_index)

        # Step in the environment
        obs, reward, done, info = env.step(action_bits)

        # Next observation -> dict -> 33D -> sliding window -> 132D
        next_event_dict = event_from_obs_gym(obs, all_types, all_attributes)
        next_state = sliding_window_encoder(next_event_dict, all_types, all_attributes)

        # One-hot encode the action
        action_one_hot = action_encoder(action_index)

        # Push transition into replay buffer
        buffer.push(state, action_one_hot, reward, next_state, done)

        # Update the current state
        state = next_state

        # Train the DQN
        loss_val = train(q_net, target_q_net, buffer, optimizer, batch_size, gamma, device=device)
        if loss_val is not None:
            training_loss.append(loss_val)

        # Evaluate periodically
        if episode % evaluation_interval == 0:
            target_q_net.load_state_dict(q_net.state_dict())
            test_env = gym.make(env_name)
            avg_reward, success_rate = evaluate_dqn(
                test_env, q_net, episodes=evaluation_episodes, device=device
            )
            average_rewards.append(avg_reward)
            average_successes.append(success_rate)
            print(f"[Gamma {gamma}] Episode {episode} - AvgReward: {avg_reward:.2f}, Success: {success_rate:.1f}%")
        
    weights_dir = Path(
        "/Users/robinguiavarch/Documents/git_projects/"
        "robot_task_recognition_rl/"
        "agent1_patterns_chests_to_reach/"
        "approach2_temporal_window/weights"
    )
    weights_dir.mkdir(parents=True, exist_ok=True)

    filename = weights_dir / f"dqn_weights_eps{eps:.2f}_gamma{gamma:.2f}.pt"
    torch.save(q_net.state_dict(), filename)

    return training_loss, average_rewards, average_successes
