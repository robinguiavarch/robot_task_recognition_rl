# agent1_patterns_chests_to_reach/approach2_temporal_window/evaluation/evaluation_actor.py
from __future__ import annotations
import torch, gym, numpy as np
from typing import Tuple

# --- dépendances internes -----------------------------------------------
from agent1_patterns_chests_to_reach.env.register_envs          import all_types, all_attributes
from agent1_patterns_chests_to_reach.utils.event_encoding       import event_to_dict_from_gym as evt_from_obs
from agent1_patterns_chests_to_reach.approach2_temporal_window.encoder.encoders_LSTM_Actor_Critic \
     import encode_symbol_bg_fg, convert_index_to_action
from agent1_patterns_chests_to_reach.approach2_temporal_window.agents.actor_lstm_agent import ActorLSTM
# -------------------------------------------------------------------------


@torch.no_grad()
def evaluate_actor(
        env: gym.Env,
        model: ActorLSTM,
        episodes: int = 50,
        device: str = "cpu",
        deterministic: bool = True,
) -> Tuple[float, float]:
    """
    • Joue `episodes` parties complètes avec la policy en mode évaluation
    • Retourne (reward_moyenne, taux_succès_%)
    """
    model.eval()
    total_r, succ = 0.0, 0

    for _ in range(episodes):
        obs   = env.reset()
        h, c  = model.init_hidden(batch_size=1, device=device)
        done  = False
        ep_r  = 0.0

        while not done:
            # ---- encode 33D
            evt   = evt_from_obs(obs, all_types, all_attributes)
            obs33 = encode_symbol_bg_fg(evt, all_types, all_attributes)
            obs_t = torch.from_numpy(obs33).float().unsqueeze(0).to(device)

            # ---- action
            a_idx, _, (h, c) = model.act(obs_t, (h, c), deterministic)
            obs, r, done, _  = env.step(convert_index_to_action(a_idx))

            ep_r += r

        total_r += ep_r
        succ    += int(ep_r >= 3.0)          

    return total_r / episodes, 100 * succ / episodes
