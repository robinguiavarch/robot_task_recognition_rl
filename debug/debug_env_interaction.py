"""
Debug script to collect and visualize events from OpenTheChests environments.

Usage:
    Run directly to visualize a stream of observed events.
"""

import gym

from agent1_patterns_chests_to_reach.env.register_envs import (
    all_types,
    all_attributes,
    register_custom_envs,
)
from agent1_patterns_chests_to_reach.utils.event_encoding import event_to_dict_from_gym
from agent1_patterns_chests_to_reach.utils.visualization import plot_event_timeline


def collect_observations(env_name, num_steps=30):
    """
    Collect a sequence of event observations from a Gym-compatible environment.

    Args:
        env_name (str): Environment name (e.g., "OpenTheChests-v1").
        num_steps (int): Number of time steps to simulate.

    Returns:
        list: List of parsed event dictionaries.
    """
    env = gym.make(env_name)
    events = []

    obs = env.reset()
    events.append(event_to_dict_from_gym(obs, all_types, all_attributes))

    for step in range(num_steps):
        action = [0] * env.action_space.n
        obs, reward, done, info = env.step(action)
        events.append(event_to_dict_from_gym(obs, all_types, all_attributes))

        if done:
            print(f"Environment ended at step {step}. Resetting.")
            obs = env.reset()

    env.close()
    return events


if __name__ == "__main__":
    register_custom_envs()
    observations = collect_observations("OpenTheChests-v1", num_steps=50)
    plot_event_timeline(observations, env_name="OpenTheChests-v1")
