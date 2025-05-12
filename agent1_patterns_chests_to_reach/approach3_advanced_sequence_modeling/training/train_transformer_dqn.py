


import os
import gym
import torch
import numpy as np
from pathlib import Path


from agent1_patterns_chests_to_reach.env.register_envs import register_custom_envs, all_types, all_attributes
from agent1_patterns_chests_to_reach.approach3_advanced_sequence_modeling.agents.replay_buffer_sliding_window import (
    ReplayBuffer,
)
from agent1_patterns_chests_to_reach.approach3_advanced_sequence_modeling.encoders.encoders_sliding_window import (
    WINDOW_SIZE,
    sliding_window_encoder,
    reset_sliding_window,
    convert_index_to_action,
)
from agent1_patterns_chests_to_reach.env.register_envs import (
    register_custom_envs,
)
from agent1_patterns_chests_to_reach.approach3_advanced_sequence_modeling.agents.transformer_dqn_agent_sliding_window import (
    TransformerDQNAgent,
)
from agent1_patterns_chests_to_reach.approach3_advanced_sequence_modeling.evaluation.evaluation_transformer_dqn import (
    evaluate_transformer_dqn,
)

save_path = Path(
        "/Users/robinguiavarch/Documents/git_projects/"
        "robot_task_recognition_rl/"
        "agent1_patterns_chests_to_reach/"
        "approach3_advanced_sequence_modeling/weights//transformer_dqn.pth"
    )

def train_transformer_dqn(
    env_id: str = "OpenTheChests-v2",
    device: str = "cpu",
    episodes: int = 500,
    max_steps: int = 200,
    buffer_size: int = 10_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-4,
    sync_freq: int = 1_000,
    eval_interval: int = 50,
    eval_episodes: int = 100,
    save_path: str = save_path,
):
    # 1) register & make env
    register_custom_envs()
    env = gym.make(env_id)

    # 2) build agent & replay buffer
    input_dim  = WINDOW_SIZE * 33
    action_dim = env.action_space.n
    agent = TransformerDQNAgent(input_dim, action_dim, device=device, gamma=gamma, lr=lr)
    buffer = ReplayBuffer(buffer_size)

    total_steps = 0
    # create dossier si besoin
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    loss_history  = []
    eval_rewards  = []
    eval_success  = []

    # 3) Training iteration
    for ep in range(1, episodes + 1):
        raw_obs = env.reset()
        reset_sliding_window()
        evt   = raw_obs if isinstance(raw_obs, dict) else raw_obs
        state = sliding_window_encoder(evt, all_types, all_attributes)

        ep_reward = 0.0
        done = False

        for t in range(max_steps):
            # a) selection + step
            a_idx  = agent.select_action(state)
            action = convert_index_to_action(a_idx)

            nxt_obs, rew, done, _ = env.step(action)
            ep_reward += rew

            nxt_evt   = nxt_obs if isinstance(nxt_obs, dict) else nxt_obs
            nxt_state = sliding_window_encoder(nxt_evt, all_types, all_attributes)

            buffer.push(state, a_idx, rew, nxt_state, done)
            state = nxt_state
            total_steps += 1

            # b) update
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                loss  = agent.update(batch)
                loss_history.append(loss)

            # c) sync target
            if total_steps % sync_freq == 0:
                agent.sync_target()

            if done:
                break

        print(f"Episode {ep:4d} | Reward {ep_reward:.1f} | ε {agent.eps:.3f}")

        # 4) periodical evaluation
        if ep % eval_interval == 0:
            torch.save(agent.q_net.state_dict(), save_path)
            mean_r, succ = evaluate_transformer_dqn(
                weight_path=str(save_path),
                env_id=env_id,
                episodes=eval_episodes,
                device=device,
            )
            print(f" → eval@{ep:4d}: reward={mean_r:.2f}, success%={succ:.1f}")
            eval_rewards.append(mean_r)
            eval_success.append(succ)

    # 5) final save
    torch.save(agent.q_net.state_dict(), save_path)
    print(f"Model finally saved to {save_path}")

    return loss_history, eval_rewards, eval_success



