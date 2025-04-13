import os
import gym
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import StepAPICompatibility

from agent1_patterns_chests_to_reach.env.register_envs import register_custom_envs
from agent1_patterns_chests_to_reach.env.wrappers import DictToMultiInputWrapper


def train_ppo(env_id: str,
              total_timesteps: int = 50000,
              save_path: str = None,
              log_path: str = None,
              learning_rate: float = 3e-4,
              policy: str = "MultiInputPolicy"):
    """
    Train a PPO agent on a custom environment with Dict observations.

    Args:
        env_id (str): Gym environment ID.
        total_timesteps (int): Number of training timesteps.
        save_path (str): Optional path to save the trained model.
        log_path (str): Optional path for monitor.csv and TensorBoard logs.
        learning_rate (float): Learning rate for optimizer.
        policy (str): Policy architecture (should be 'MultiInputPolicy').

    Returns:
        PPO: Trained PPO model.
    """
    register_custom_envs()

    # Create and wrap the environment
    env = gym.make(env_id)
    env = StepAPICompatibility(env, output_truncation_bool=True)  # ✅ Converts step() to 5-tuple
    env = Monitor(env, log_path)                                   # ✅ For logging rewards and episode length
    env = DictToMultiInputWrapper(env)                             # ✅ To support Dict obs with MultiInputPolicy

    # Initialize and train PPO agent
    model = PPO(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        verbose=1,
        tensorboard_log=log_path
    )

    model.learn(total_timesteps=total_timesteps)

    if save_path:
        model.save(save_path)
        print(f"✅ Model saved to {save_path}")

    return model
