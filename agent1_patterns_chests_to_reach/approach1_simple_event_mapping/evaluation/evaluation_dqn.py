# evaluation_dqn.py

import torch
import numpy as np
from agent1_patterns_chests_to_reach.approach1_simple_event_mapping.encoder.encoders import encode_symbol_bg_fg, convert_index_to_action
from agent1_patterns_chests_to_reach.utils.event_encoding import event_to_dict_from_gym as event_from_obs_gym
from agent1_patterns_chests_to_reach.env.register_envs import all_types, all_attributes

def evaluate_dqn(env, q_net, episodes=20, device="cpu", verbose=False):
    """
    Evaluate a DQN policy over a number of episodes.

    Args:
        env: Gym environment.
        q_net: Trained Q-network.
        episodes (int): Number of episodes to run for evaluation.
        device (str): Torch device.
        verbose (bool): If True, prints step-by-step events.

    Returns:
        float: Mean reward over episodes.
        float: Success rate (reward > 0).
    """
    q_net.eval()  # set the network to eval mode
    total_rewards = []
    successes = 0

    for ep in range(episodes):
        # 1) Reset and encode the initial observation
        obs = env.reset()
        event_dict = event_from_obs_gym(obs, all_types, all_attributes)
        state = encode_symbol_bg_fg(event_dict, all_types, all_attributes)

        done = False
        ep_reward = 0.0

        while not done:
            # 2) Convert state to tensor and get Q-values
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = q_net(state_tensor)
                action_index = torch.argmax(q_values, dim=-1).item()
            
            # 3) Convert action index to 3-bit action
            action = convert_index_to_action(action_index)
            
            # 4) Step in environment
            obs, reward, done, _ = env.step(action)
            
            # 5) Encode next state
            event_dict = event_from_obs_gym(obs, all_types, all_attributes)
            state = encode_symbol_bg_fg(event_dict, all_types, all_attributes)
            
            ep_reward += reward

            if verbose:
                print(f"[EP {ep}] Action index: {action_index}, Action: {action}, Reward: {reward}")

        total_rewards.append(ep_reward)
        if ep_reward == 3:
            successes += 1

    mean_reward = np.mean(total_rewards)
    success_rate = (successes / episodes) * 100

    return mean_reward, success_rate