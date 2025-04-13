import gym
import argparse
import numpy as np
import yaml
import os

from stable_baselines3 import PPO, DQN
from agent1_patterns_chests_to_reach.env.register_envs import register_custom_envs
from agent1_patterns_chests_to_reach.utils.event_encoding import event_to_dict_from_gym


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate(model, env, n_episodes=20, render=False):
    total_rewards = []
    successes = 0

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if render:
                event = event_to_dict_from_gym(obs, env.types, env.attributes)
                print(f"[Step] Event={event['symbol']}, Action={action}, Reward={reward}")

        total_rewards.append(ep_reward)
        if ep_reward > 0:
            successes += 1

    avg_reward = np.mean(total_rewards)
    success_rate = (successes / n_episodes) * 100
    return avg_reward, success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--render", action="store_true", help="Print step-by-step evaluation")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Register environment
    register_custom_envs()
    env = gym.make(cfg["env_id"])

    # Load the right model
    algo = cfg["algorithm"]
    if algo == "PPO":
        model = PPO.load(cfg["save_path"])
    elif algo == "DQN":
        model = DQN.load(cfg["save_path"])
    else:
        raise ValueError(f"Unsupported algorithm for evaluation: {algo}")

    # Run evaluation
    avg_reward, success_rate = evaluate(model, env, n_episodes=20, render=args.render)

    print(f"\n🎯 Evaluation Results — {algo}")
    print(f"Avg Reward:     {avg_reward:.2f}")
    print(f"Success Rate:   {success_rate:.1f}%")
