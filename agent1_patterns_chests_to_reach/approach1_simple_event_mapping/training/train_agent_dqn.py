import gym
import torch
import numpy as np
from torch.optim import Adam
from pathlib import Path

from agent1_patterns_chests_to_reach.env.register_envs import (
    register_custom_envs,
    all_types,
    all_attributes,
)
from agent1_patterns_chests_to_reach.approach1_simple_event_mapping.encoder.encoders import (
    encode_symbol_bg_fg,
    action_encoder,
    convert_index_to_action,
)
from agent1_patterns_chests_to_reach.utils.event_encoding import (
    event_to_dict_from_gym as event_from_obs_gym,
)
from agent1_patterns_chests_to_reach.approach1_simple_event_mapping.agents.replay_buffer import ReplayBuffer
from agent1_patterns_chests_to_reach.approach1_simple_event_mapping.evaluation.evaluation_dqn import evaluate_dqn
from agent1_patterns_chests_to_reach.approach1_simple_event_mapping.agents.dqn_agent import (
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
    Trains a DQN agent on the specified environment (e.g. "OpenTheChests-v0").
    We use encode_symbol_bg_fg(...) to encode each observation into a 33D vector
    (15 symbols + 8 bg + 8 fg + 2 times = 33 dimensions).

    Args:
        env_name (str): Name of the Gym environment (e.g. "OpenTheChests-v0").
        buffer_capacity (int): Maximum number of transitions in the replay buffer.
        epochs (int): Number of training episodes.
        batch_size (int): Batch size for each training step.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor.
        evaluation_interval (int): Evaluate the agent every 'evaluation_interval' episodes.
        evaluation_episodes (int): Number of episodes used for each evaluation.
        eps (float): Epsilon for epsilon-greedy exploration.

    Returns:
        tuple (training_loss, eval_rewards, success_rates):
            - training_loss: list of loss values (one per training step).
            - eval_rewards: list of average rewards (one per evaluation).
            - success_rates: list of success rates (one per evaluation).
    """

    # 1) Register custom environments
    register_custom_envs()

    # 2) Choose the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) Create the environment
    env = gym.make(env_name)

    # We know each observation is encoded into 33 dimensions (symbol + colors + times).
    state_size = 33
    action_size = 8  # 3-bit actions → 2^3 = 8

    # Instantiate main and target QNetworks
    q_net = QNetwork(state_size, action_size).to(device)
    target_q_net = QNetwork(state_size, action_size).to(device)
    target_q_net.load_state_dict(q_net.state_dict())

    # 4) Optimizer and replay buffer
    optimizer = Adam(q_net.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(buffer_capacity)

    # 5) Initial observation and encoding
    obs = env.reset()
    # Convert the raw Gym observation into a dictionary
    event_dict = event_from_obs_gym(obs, all_types, all_attributes)
    # Encode this dictionary into a 33D vector
    state = encode_symbol_bg_fg(event_dict, all_types, all_attributes)
    done = False

    training_loss = []
    average_rewards = []
    average_successes = []

    for episode in range(epochs):
        if done:
            # Reset the environment and re-encode the initial observation
            obs = env.reset()
            event_dict = event_from_obs_gym(obs, all_types, all_attributes)
            state = encode_symbol_bg_fg(event_dict, all_types, all_attributes)
            done = False

        # Epsilon-greedy action selection
        action_index = select_action(q_net, state, eps=eps, device=device)

        # Convert the action index (0..7) to a 3-bit vector [0..1, 0..1, 0..1]
        action_bits = np.array([int(digit) for digit in bin(action_index).removeprefix("0b").zfill(3)])

        # Step in the environment using the 3-bit action
        obs, reward, done, info = env.step(action_bits)

        # Encode the new state (next observation)
        next_event_dict = event_from_obs_gym(obs, all_types, all_attributes)
        next_state = encode_symbol_bg_fg(next_event_dict, all_types, all_attributes)

        # Convert the action index to a one-hot 8D vector
        action_one_hot = action_encoder(action_index)

        # Store the transition in the replay buffer
        buffer.push(state, action_one_hot, reward, next_state, done)

        # Train the Q-network
        loss_val = train(q_net, target_q_net, buffer, optimizer, batch_size, gamma, device=device)
        if loss_val is not None:
            training_loss.append(loss_val)

        # Move on to the next state
        state = next_state

        # Periodic evaluation
        if episode % evaluation_interval == 0:
            # Sync target network
            target_q_net.load_state_dict(q_net.state_dict())

            # Evaluate the agent on 'evaluation_episodes'
            test_env = gym.make(env_name)
            avg_reward, success_rate = evaluate_dqn(
                test_env, q_net, episodes=evaluation_episodes, device=device
            )
            average_rewards.append(avg_reward)
            average_successes.append(success_rate)

            print(
                f"[Gamma {gamma}] Episode {episode} - "
                f"AvgReward: {avg_reward:.2f}, Success: {success_rate:.1f}%"
            )

    weights_dir = Path(
        "/Users/robinguiavarch/Documents/git_projects/"
        "robot_task_recognition_rl/"
        "agent1_patterns_chests_to_reach/"
        "approach1_simple_event_mapping/weights"
    )
    weights_dir.mkdir(parents=True, exist_ok=True)

    filename = weights_dir / f"dqn_weights_eps{eps:.2f}_gamma{gamma:.2f}.pt"
    torch.save(q_net.state_dict(), filename)

    # Return the lists for plotting or further analysis
    return training_loss, average_rewards, average_successes