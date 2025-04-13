# evaluation_dqn.py

import torch
import numpy as np
from agent1_patterns_chests_to_reach.utils.encoders import simple_encoder, convert_index_to_action
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
    q_net.eval()  # set the network to evaluation mode
    total_rewards = []
    successes = 0

    for ep in range(episodes):
        obs = env.reset()
        state = simple_encoder(event_from_obs_gym(obs, all_types, all_attributes))
        done = False
        ep_reward = 0.0

        while not done:
            # Convert state to tensor and get Q-values
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = q_net(state_tensor)
                action_index = torch.argmax(q_values, dim=-1).item()
            
            # Convert index to real action
            action = convert_index_to_action(action_index)
            
            # Step in environment
            obs, reward, done, _ = env.step(action)
            
            # Encode next state
            state = simple_encoder(event_from_obs_gym(obs, all_types, all_attributes))
            
            ep_reward += reward

            if verbose:
                print(f"[EP {ep}] Action index: {action_index}, Action: {action}, Reward: {reward}")

        total_rewards.append(ep_reward)
        if ep_reward > 0:
            successes += 1

    mean_reward = np.mean(total_rewards)
    success_rate = (successes / episodes) * 100

    return mean_reward, success_rate
