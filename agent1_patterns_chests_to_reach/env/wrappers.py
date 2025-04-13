import gym
import numpy as np
from gym import spaces


class DictToMultiInputWrapper(gym.ObservationWrapper):
    """
    Observation wrapper that ensures observations in Dict spaces are properly formatted
    for use with SB3's MultiInputPolicy.

    This wrapper converts each item in the dictionary observation to a NumPy array,
    ensuring compatibility with stable-baselines3 input format.
    """

    def __init__(self, env):
        super().__init__(env)

        assert isinstance(env.observation_space, spaces.Dict), (
            f"Expected Dict observation space, but got {type(env.observation_space)}"
        )

        self.observation_space = env.observation_space

    def observation(self, observation):
        return {
            key: np.array(value, dtype=np.int32)
            for key, value in observation.items()
        }
