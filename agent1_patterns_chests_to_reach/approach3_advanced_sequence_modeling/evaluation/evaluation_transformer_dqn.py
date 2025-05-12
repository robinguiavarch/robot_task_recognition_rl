# evaluation_transformer_dqn.py

import torch
import numpy as np
import gym

from agent1_patterns_chests_to_reach.env.register_envs import register_custom_envs, all_types, all_attributes
from agent1_patterns_chests_to_reach.approach3_advanced_sequence_modeling.agents.transformer_dqn_agent_sliding_window import TransformerDQNAgent
from agent1_patterns_chests_to_reach.approach3_advanced_sequence_modeling.encoders.encoders_sliding_window import (
    WINDOW_SIZE, sliding_window_encoder, reset_sliding_window, convert_index_to_action,
)

def evaluate_transformer_dqn(
    weight_path: str,
    env_id: str = "OpenTheChests-v2",
    episodes: int = 100,
    device: str = "cpu",
):
    # 1) register & make env
    register_custom_envs()
    env = gym.make(env_id)

    # 2) rebuild agent + load weights
    input_dim  = WINDOW_SIZE * 33
    action_dim = env.action_space.n
    agent = TransformerDQNAgent(input_dim, action_dim, device=device)
    agent.q_net.load_state_dict(torch.load(weight_path, map_location=device))
    agent.q_net.eval()

    total_rewards = []
    successes     = 0

    # 3) rollouts
    for _ in range(episodes):
        raw_obs = env.reset()
        reset_sliding_window()
        state = sliding_window_encoder(raw_obs, all_types, all_attributes)

        done = False
        ep_r = 0.0
        while not done:
            a_idx = agent.select_action(state)
            action = convert_index_to_action(a_idx)
            nxt_obs, rew, done, _ = env.step(action)
            ep_r += rew
            state = sliding_window_encoder(nxt_obs, all_types, all_attributes)

        total_rewards.append(ep_r)
        if ep_r > 0:
            successes += 1

    mean_reward  = float(np.mean(total_rewards))
    success_rate = float(successes / episodes * 100)
    return mean_reward, success_rate



