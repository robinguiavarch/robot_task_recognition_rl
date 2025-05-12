import torch
import numpy as np
from agent1_patterns_chests_to_reach.approach1_simple_event_mapping.encoder.encoders import (
    encode_symbol_bg_fg,
    convert_index_to_action,
)
from agent1_patterns_chests_to_reach.utils.event_encoding import (
    event_to_dict_from_gym as event_from_obs_gym,
)
from agent1_patterns_chests_to_reach.env.register_envs import all_types, all_attributes


def evaluate_actor_critic(env, actor, episodes=20, device="cpu", verbose=False):
    """
    Evaluate an Actor-Critic agent with Q-value critic over a number of episodes.

    Args:
        env: Gym environment.
        actor: Trained actor network (outputs π(a|s)).
        episodes (int): Number of episodes to run for evaluation.
        device (str): Torch device.
        verbose (bool): If True, prints step-by-step events.

    Returns:
        float: Mean reward over episodes.
        float: Success rate (% episodes with reward == 3).
    """
    actor.eval()
    total_rewards = []
    successes = 0

    for ep in range(episodes):
        obs = env.reset()
        event_dict = event_from_obs_gym(obs, all_types, all_attributes)
        state = encode_symbol_bg_fg(event_dict, all_types, all_attributes)
        done = False
        ep_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = actor(state_tensor)               # π(a|s)
                action_index = torch.argmax(probs, dim=-1).item()
                action = convert_index_to_action(action_index)

            obs, reward, done, _ = env.step(action)
            event_dict = event_from_obs_gym(obs, all_types, all_attributes)
            state = encode_symbol_bg_fg(event_dict, all_types, all_attributes)
            ep_reward += reward

            if verbose:
                print(f"[EP {ep}] Action: {action}, Reward: {reward}")

        total_rewards.append(ep_reward)
        successes += int(ep_reward == 3)

    return np.mean(total_rewards), (successes / episodes) * 100
