import gym
import torch
from torch.optim import Adam
from pathlib import Path
import os

from agent1_patterns_chests_to_reach.env.register_envs import (
    register_custom_envs,
    all_types,
    all_attributes,
)
from agent1_patterns_chests_to_reach.approach1_simple_event_mapping.encoder.encoders import (
    encode_symbol_bg_fg,
    convert_index_to_action,
)
from agent1_patterns_chests_to_reach.utils.event_encoding import (
    event_to_dict_from_gym as event_from_obs_gym,
)
from agent1_patterns_chests_to_reach.approach1_simple_event_mapping.agents.actor_critic import (
    Actor,
    CriticQ,
    select_action,
    update,
)
from agent1_patterns_chests_to_reach.approach1_simple_event_mapping.evaluation.evaluation_actor_critic import (
    evaluate_actor_critic,
)


def train_actor_critic(
    env_name="OpenTheChests-v0",
    gamma=0.99,
    episodes=500,
    lr_actor=1e-3,
    lr_critic=1e-3,
    eval_interval=50,
    eval_episodes=50,
    device=None
):
    register_custom_envs()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_name)
    state_size = 33
    action_size = 8

    actor = Actor(state_size, action_size).to(device)
    critic = CriticQ(state_size, action_size).to(device)
    optimizer_actor = Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = Adam(critic.parameters(), lr=lr_critic)

    training_actor_losses = []
    training_critic_losses = []
    eval_rewards = []
    success_rates = []


    
    for episode in range(episodes):
        obs = env.reset()
        event_dict = event_from_obs_gym(obs, all_types, all_attributes)
        state = encode_symbol_bg_fg(event_dict, all_types, all_attributes)

        states, actions, log_probs, rewards = [], [], [], []
        next_states, dones = [], []
        done = False

        while not done:
            action_index, log_prob = select_action(actor, state)
            action = convert_index_to_action(action_index)

            obs, reward, done, _ = env.step(action)
            next_event_dict = event_from_obs_gym(obs, all_types, all_attributes)
            next_state = encode_symbol_bg_fg(next_event_dict, all_types, all_attributes)

            states.append(torch.tensor(state, dtype=torch.float32))
            actions.append(action_index)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
            next_states.append(torch.tensor(next_state, dtype=torch.float32))
            dones.append(done)

        actor_loss, critic_loss = update(
            actor, critic, optimizer_actor, optimizer_critic,
            states, actions, log_probs, rewards, next_states, dones, gamma
        )
        training_actor_losses.append(actor_loss)
        training_critic_losses.append(critic_loss)

        if (episode + 1) % eval_interval == 0:
            avg_reward, success_rate = evaluate_actor_critic(
                gym.make(env_name), actor,
                episodes=eval_episodes,
                device=device,
                verbose=False
            )
            eval_rewards.append(avg_reward)
            success_rates.append(success_rate)

            print(f"[Ep {episode+1}] Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f} | "
                  f"EvalR: {avg_reward:.2f} | Success: {success_rate:.1f}%")

    weights_dir = Path(
        "/Users/robinguiavarch/Documents/git_projects/"
        "robot_task_recognition_rl/"
        "agent1_patterns_chests_to_reach/"
        "approach1_simple_event_mapping/weights"
    )
    weights_dir.mkdir(parents=True, exist_ok=True)

    actor_path  = weights_dir / f"ac_actor_gamma{gamma:.2f}.pt"
    critic_path = weights_dir / f"ac_critic_gamma{gamma:.2f}.pt"
    torch.save(actor.state_dict(),  actor_path)
    torch.save(critic.state_dict(), critic_path)
    print(f" Saved actor → {actor_path}")
    print(f" Saved critic → {critic_path}")

    env.close()
    return training_actor_losses, training_critic_losses, eval_rewards, success_rates


