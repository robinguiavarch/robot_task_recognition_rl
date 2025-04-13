# agent1_patterns_chests_to_reach/env/vec_transpose_dict.py

from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper


class VecTransposeDict(VecEnvWrapper):
    """
    Transposes the observation dictionary so that it is compatible
    with MultiInputPolicy.
    """

    def __init__(self, venv):
        super().__init__(venv)
        assert isinstance(venv.observation_space, dict) or hasattr(venv.observation_space, "spaces"), \
            "VecTransposeDict only works with dict-style observation spaces."

    def reset(self):
        obs = self.venv.reset()
        return self.transpose(obs)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return self.transpose(obs), reward, done, info

    def transpose(self, obs_dict):
        # Transpose obs_dict: dict of batch → batch of dict
        return {key: [obs[key] for obs in obs_dict] for key in obs_dict[0]}
