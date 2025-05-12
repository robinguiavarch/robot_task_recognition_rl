# agent1_patterns_chests_to_reach/approach2_temporal_window/agents/recurrent_ppo_agent.py
from __future__ import annotations
import gym
import numpy as np
from gym import Wrapper
from gym.spaces import Box
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agent1_patterns_chests_to_reach.env.register_envs import (
    register_custom_envs, all_types, all_attributes
)
from agent1_patterns_chests_to_reach.utils.event_encoding import event_to_dict_from_gym
from agent1_patterns_chests_to_reach.approach2_temporal_window.encoder.encoders_LSTM_Actor_Critic import encode_symbol_bg_fg

register_custom_envs()

class EncodeObsWrapper(Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(33,), dtype=np.float32)

    @staticmethod
    def _encode_obs(raw_obs) -> np.ndarray:
        evt = event_to_dict_from_gym(raw_obs, all_types, all_attributes)
        vec = encode_symbol_bg_fg(evt, all_types, all_attributes)
        return vec.astype(np.float32)

    def reset(self, **kwargs):
        raw_obs = self.env.reset(**kwargs)
        obs_enc = self._encode_obs(raw_obs)
        return obs_enc  # legacy Gym: pas d'info dict

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs_enc = self._encode_obs(raw_obs)
        return obs_enc, reward, done, info  # ← legacy Gym, exactement 4 éléments


def make_env(env_id: str):
    def _init():
        register_custom_envs()
        base_env = gym.make(env_id)
        return EncodeObsWrapper(base_env)
    return DummyVecEnv([_init])


def make_recurrent_ppo(env_id: str, **model_kwargs) -> RecurrentPPO:
    env = make_env(env_id)
    return RecurrentPPO("MlpLstmPolicy", env, verbose=1, **model_kwargs)
