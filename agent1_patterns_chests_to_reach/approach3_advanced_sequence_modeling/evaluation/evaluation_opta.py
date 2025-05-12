# agent1_patterns_chests_to_reach/approach3_advanced_sequence_modeling/evaluation/evaluation_opta.py

from __future__ import annotations
import torch, gym, numpy as np
from typing import Tuple

# --- dépendances internes -----------------------------------------------
from agent1_patterns_chests_to_reach.env.register_envs import all_types, all_attributes
from agent1_patterns_chests_to_reach.utils.event_encoding import event_to_dict_from_gym as evt_from_obs
from agent1_patterns_chests_to_reach.approach3_advanced_sequence_modeling.encoders.encoder_opta import (
    encode_symbol_bg_fg,
    convert_index_to_action,
)
from agent1_patterns_chests_to_reach.approach3_advanced_sequence_modeling.agents.opta_agent import OPTransformer
# ------------------------------------------------------------------------


@torch.no_grad()
def evaluate_opta(
    env: gym.Env,
    model: OPTransformer,
    episodes: int = 50,
    device: str = "cpu",
    deterministic: bool = True,
) -> Tuple[float, float]:
    """
    • Run `episodes` full rollouts in eval mode with the policy.
    • Returns (average_reward, success_rate_%)
    """
    model.eval()
    total_r, succ = 0.0, 0

    for _ in range(episodes):
        obs = env.reset()
        model.reset_episode()
        done = False
        ep_r = 0.0

        while not done:
            evt = evt_from_obs(obs, all_types, all_attributes)
            obs_vec = encode_symbol_bg_fg(evt, all_types, all_attributes)

            a_idx = model.act_eval(obs_vec, deterministic=deterministic, device=device)
            obs, r, done, _ = env.step(convert_index_to_action(a_idx))
            ep_r += r

        total_r += ep_r
        succ += int(ep_r >= 3.0)

    return total_r / episodes, 100 * succ / episodes
