
import gym, torch, numpy as np
from torch.optim import Adam
from typing import List
from pathlib import Path

# --- project imports -----------------------------------------------------
from agent1_patterns_chests_to_reach.env.register_envs import register_custom_envs, all_types, all_attributes
from agent1_patterns_chests_to_reach.utils.event_encoding import event_to_dict_from_gym as evt_from_obs
from agent1_patterns_chests_to_reach.approach3_advanced_sequence_modeling.encoders.encoder_opta import encode_symbol_bg_fg, convert_index_to_action
from agent1_patterns_chests_to_reach.approach3_advanced_sequence_modeling.agents.opta_agent import OPTransformer
from agent1_patterns_chests_to_reach.approach3_advanced_sequence_modeling.evaluation.evaluation_opta import evaluate_opta
# -------------------------------------------------------------------------

def compute_mc_returns(rewards: List[float], gamma: float, device: torch.device):
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32, device=device)

def train_opta(
    env_name: str = "OpenTheChests-v2",
    episodes: int = 10_000,
    gamma: float = 0.99,
    lr: float = 3e-4,
    entropy_coef: float = 1e-2,
    grad_clip: float = 0.5,
    eval_every: int = 200,
    eval_episodes: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    # ---------------- init env + modèle ----------------
    register_custom_envs()
    env = gym.make(env_name)
    model = OPTransformer().to(device)
    optim = Adam(model.parameters(), lr=lr)

    policy_losses, entropies, eval_R, succ_R = [], [], [], []

    for ep in range(episodes):
        model.reset_episode()
        obs = env.reset()
        done = False

        log_probs, rewards, entropies_t = [], [], []

        # ---------- generate full episode ----------
        while not done:
            evt = evt_from_obs(obs, all_types, all_attributes)
            obs_vec = encode_symbol_bg_fg(evt, all_types, all_attributes)

            a_idx, log_p, probs = model.act(obs_vec, deterministic=False, device=device)

            # Compute entropy at timestep
            ent = -torch.sum(probs * torch.log(probs + 1e-8))
            entropies_t.append(ent)

            obs, r, done, _ = env.step(convert_index_to_action(a_idx))
            log_probs.append(log_p)
            rewards.append(float(r))

        # ---------- compute returns ----------
        returns = compute_mc_returns(rewards, gamma, device)
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        log_probs = torch.stack(log_probs)            # (T,)
        entropy = torch.stack(entropies_t).mean()     # (scalar)

        loss_pg = -(log_probs * returns).mean()
        loss = loss_pg - entropy_coef * entropy

        optim.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        policy_losses.append(loss.item())
        entropies.append(entropy.item())

        # ---------------- evaluation -----------------
        if (ep + 1) % eval_every == 0:
            avg_r, sr = evaluate_opta(gym.make(env_name), model, eval_episodes, device=device)
            eval_R.append(avg_r)
            succ_R.append(sr)
            print(f"[Ep {ep+1}] loss {loss.item():.3f} | "
                  f"trainR {np.sum(rewards):.2f} | evalR {avg_r:.2f} | success {sr:.1f}%")

    weights_dir = Path(
        "/Users/robinguiavarch/Documents/git_projects/"
        "robot_task_recognition_rl/"
        "agent1_patterns_chests_to_reach/"
        "approach3_advanced_sequence_modeling/weights"
    )
    weights_dir.mkdir(parents=True, exist_ok=True)

    model_path  = weights_dir / f"opta{gamma:.2f}.pt"
    torch.save(model.state_dict(),  model_path)

    print(f" Saved opta → {model_path}")


    return model, policy_losses, entropies, eval_R, succ_R
