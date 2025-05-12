import torch
import numpy as np
from agent1_patterns_chests_to_reach.approach2_temporal_window.encoder.encoders_sliding_window import (
    sliding_window_encoder,
    convert_index_to_action,
    reset_sliding_window,  # clears the 4-event history
)
from agent1_patterns_chests_to_reach.utils.event_encoding import event_to_dict_from_gym as event_from_obs_gym
from agent1_patterns_chests_to_reach.env.register_envs import all_types, all_attributes

def evaluate_dqn(env, q_net, episodes=20, device="cpu", verbose=False):
    """
    Evaluate a DQN policy over a number of episodes using a sliding-window encoder.

    Each event is encoded into a 33D vector (symbol + bg/fg colors + times),
    and we keep a window of up to 4 events, resulting in a 132D state.

    Args:
        env: The Gym environment.
        q_net: Trained Q-network (expects 132D if 4×33).
        episodes (int): Number of episodes for evaluation.
        device (str): Torch device ("cpu" or "cuda").
        verbose (bool): If True, prints step details.

    Returns:
        float: Mean reward over these episodes.
        float: Success rate (percentage of episodes with reward > 0).
    """
    q_net.eval()
    total_rewards = []
    successes = 0

    for ep in range(episodes):
        # Reset environment and the sliding window
        obs = env.reset()
        reset_sliding_window()

        # Convert initial observation to 33D event -> 132D state
        event_dict = event_from_obs_gym(obs, all_types, all_attributes)
        state = sliding_window_encoder(event_dict, all_types, all_attributes)

        done = False
        ep_reward = 0.0

        while not done:
            # Convert state to tensor and get Q-values
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = q_net(state_tensor)
                action_index = torch.argmax(q_values, dim=-1).item()
            
            # Convert the selected action index (0..7) to a 3-bit action
            action = convert_index_to_action(action_index)
            
            # Step in the environment
            obs, reward, done, _ = env.step(action)
            
            # Encode the new observation into the sliding window
            event_dict = event_from_obs_gym(obs, all_types, all_attributes)
            state = sliding_window_encoder(event_dict, all_types, all_attributes)
            
            ep_reward += reward
            if verbose:
                print(f"[Eval EP {ep}] Action index: {action_index}, Action: {action}, Reward: {reward}")

        total_rewards.append(ep_reward)
        if ep_reward > 0:
            successes += 1

    mean_reward = np.mean(total_rewards)
    success_rate = (successes / episodes) * 100

    return mean_reward, success_rate
