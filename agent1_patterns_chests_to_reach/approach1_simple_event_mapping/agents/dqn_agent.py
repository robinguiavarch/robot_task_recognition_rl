from stable_baselines3 import DQN
from agent1_patterns_chests_to_reach.env.register_envs import register_custom_envs
import gym


def train_dqn(env_id: str,
              total_timesteps: int = 50000,
              save_path: str = None,
              log_path: str = None,
              learning_rate: float = 1e-3):
    """
    Train a DQN agent on the specified environment.

    Args:
        env_id (str): Gym environment ID (e.g., 'OpenTheChests-v0').
        total_timesteps (int): Number of training steps.
        save_path (str): Path to save the trained model.
        log_path (str): Path for tensorboard logging.
        learning_rate (float): Learning rate.

    Returns:
        DQN: Trained model.
    """
    register_custom_envs()
    env = gym.make(env_id)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        verbose=1,
        tensorboard_log=log_path
    )

    model.learn(total_timesteps=total_timesteps)

    if save_path:
        model.save(save_path)
        print(f"✅ DQN model saved to {save_path}")

    return model
