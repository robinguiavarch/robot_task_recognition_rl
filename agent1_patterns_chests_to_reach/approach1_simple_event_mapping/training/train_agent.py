import argparse
import yaml
import os

from agent1_patterns_chests_to_reach.approach1_simple_event_mapping.agents.ppo_agent import train_ppo
from agent1_patterns_chests_to_reach.approach1_simple_event_mapping.agents.dqn_agent import train_dqn


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Command-line argument for config file
    parser = argparse.ArgumentParser(description="Train an RL agent for event-to-action mapping.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Ensure directories exist
    os.makedirs(os.path.dirname(cfg["save_path"]), exist_ok=True)
    os.makedirs(cfg["log_path"], exist_ok=True)

    # Select and run training function
    algo = cfg["algorithm"]

    if algo == "PPO":
        train_ppo(
            env_id=cfg["env_id"],
            total_timesteps=cfg["total_timesteps"],
            save_path=cfg["save_path"],
            log_path=cfg["log_path"],
            learning_rate=cfg["learning_rate"],
            policy=cfg["policy"]
        )

    elif algo == "DQN":
        train_dqn(
            env_id=cfg["env_id"],
            total_timesteps=cfg["total_timesteps"],
            save_path=cfg["save_path"],
            log_path=cfg["log_path"],
            learning_rate=cfg["learning_rate"],
        )

    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
