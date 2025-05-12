# agent1_patterns_chests_to_reach/approach2_temporal_window/training/train_agent_actor_lstm.py

from __future__ import annotations
import gym
import torch
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from typing import List, Tuple
from pathlib import Path

# project imports
from agent1_patterns_chests_to_reach.env.register_envs \
    import register_custom_envs, all_types, all_attributes
from agent1_patterns_chests_to_reach.utils.event_encoding \
    import event_to_dict_from_gym as evt_from_obs
from agent1_patterns_chests_to_reach.approach2_temporal_window.encoder.encoders_LSTM_Actor_Critic \
    import encode_symbol_bg_fg, convert_index_to_action
from agent1_patterns_chests_to_reach.approach2_temporal_window.agents.actor_lstm_agent \
    import ActorLSTM
from agent1_patterns_chests_to_reach.approach2_temporal_window.evaluation.evaluation_actor \
    import evaluate_actor


def compute_mc_returns(rewards: List[float], gamma: float, device: torch.device) -> torch.Tensor:
    """
    Monte-Carlo returns R̂_t = Σ_{k=t}^{T-1} γ^{k-t} r_k
    Returns a tensor of shape (T,) in forward order.
    """
    R = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32, device=device)


def train_agent_actor_lstm(
    env_name      : str    = "OpenTheChests-v1",
    episodes      : int    = 10_000,
    gamma         : float  = 0.99,
    lr            : float  = 3e-4,
    entropy_coef  : float  = 1e-2,
    grad_clip     : float  = 0.5,
    rollout_len   : int    = 128,        # ← truncated rollout length
    eval_every    : int    = 200,
    eval_episodes : int    = 50,
    device        : str    = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[ActorLSTM, List[float], List[float], List[float], List[float]]:
    """
    Train an Actor‐only LSTM using truncated rollouts of length `rollout_len`.
    Updates happen every `rollout_len` environment steps.
    """
    register_custom_envs()
    env   = gym.make(env_name)
    model = ActorLSTM(obs_dim=33, action_dim=8).to(device)
    optim = Adam(model.parameters(), lr=lr)

    policy_losses: List[float] = []
    entropies:     List[float] = []
    eval_R:        List[float] = []
    succ_R:        List[float] = []

    for ep in range(1, episodes+1):
        obs = env.reset()
        h, c = model.init_hidden(batch_size=1, device=device)
        done = False

        # buffers for truncated‐rollout
        buf_logps:    List[torch.Tensor] = []
        buf_rewards:  List[float]        = []

        while not done:
            # encode
            evt    = evt_from_obs(obs, all_types, all_attributes)
            vec33  = encode_symbol_bg_fg(evt, all_types, all_attributes)
            s_t    = torch.from_numpy(vec33).float().unsqueeze(0).to(device)

            # forward
            logits, (h, c) = model(s_t, (h, c))
            h, c = h.detach(), c.detach()
            probs = torch.softmax(logits.squeeze(1), dim=-1)
            m     = torch.distributions.Categorical(probs)
            a_idx = m.sample()
            log_p = m.log_prob(a_idx)

            # step
            obs, r, done, _ = env.step(convert_index_to_action(int(a_idx)))
            buf_logps.append(log_p.squeeze())
            buf_rewards.append(float(r))

            # flush & update every rollout_len steps
            if len(buf_logps) >= rollout_len:
                returns = compute_mc_returns(buf_rewards, gamma, device)
                # returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
                logp_t  = torch.stack(buf_logps)                   # (rollout_len,)
                # entropy from last-step distribution, averaged
                ent = (-probs * torch.log(probs + 1e-8)).sum() / rollout_len

                loss = -(logp_t * returns).mean() - entropy_coef * ent

                optim.zero_grad()
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()


                # clear buffers
                buf_logps.clear()
                buf_rewards.clear()

                policy_losses.append(loss.item())
                entropies.append(ent.item())

        # if any leftover steps at episode end
        if buf_logps:
            returns = compute_mc_returns(buf_rewards, gamma, device)
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
            logp_t  = torch.stack(buf_logps)
            ent = (-probs * torch.log(probs + 1e-8)).sum() / len(buf_rewards)
            loss = -(logp_t * returns).mean() - entropy_coef * ent

            optim.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

            h, c = h.detach(), c.detach()
            policy_losses.append(loss.item())
            entropies.append(ent.item())

        # periodic evaluation
        if ep % eval_every == 0:
            avg_r, sr = evaluate_actor(gym.make(env_name),
                                       model,
                                       episodes=eval_episodes,
                                       device=device)
            eval_R.append(avg_r)
            succ_R.append(sr)
            print(f"[Ep {ep:5d}] EvalR {avg_r:.2f} | Success {sr:.1f}%")

    # (optional) save weights
    wdir = Path(__file__).parent / "weights_actor_lstm"
    wdir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), wdir / f"actor_lstm_gamma{gamma:.2f}.pt")

    return model, policy_losses, entropies, eval_R, succ_R
