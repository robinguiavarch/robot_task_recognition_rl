import gym
import torch
from torch.optim import Adam

from agent1_patterns_chests_to_reach.env.register_envs import (
    register_custom_envs,
    all_types,
    all_attributes,
)
from agent1_patterns_chests_to_reach.utils.encoders import (
    simple_encoder,
    convert_index_to_action,
)
from agent1_patterns_chests_to_reach.utils.event_encoding import (
    event_to_dict_from_gym as event_from_obs_gym,
)
from agent1_patterns_chests_to_reach.approach1_simple_event_mapping.agents.actor_critic import (
    Actor,
    Critic,
    select_action,
    update,
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
    state_size = 4
    action_size = 8

    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size).to(device)
    optimizer_actor = Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = Adam(critic.parameters(), lr=lr_critic)

    training_actor_losses = []
    training_critic_losses = []
    eval_rewards = []
    success_rates = []

    for episode in range(episodes):
        obs = env.reset()
        state = simple_encoder(event_from_obs_gym(obs, all_types, all_attributes))

        states, actions, log_probs, rewards = [], [], [], []
        done = False

        while not done:
            action_index, log_prob = select_action(actor, state, device=device)
            action = convert_index_to_action(action_index)

            obs, reward, done, _ = env.step(action)
            next_state = simple_encoder(event_from_obs_gym(obs, all_types, all_attributes))

            states.append(state)
            actions.append(action_index)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

        actor_loss, critic_loss = update(
            actor, critic, optimizer_actor, optimizer_critic,
            states, actions, log_probs, rewards, gamma, device
        )

        training_actor_losses.append(actor_loss)
        training_critic_losses.append(critic_loss)

        if episode % eval_interval == 0:
            total_reward = sum(rewards)
            success = total_reward > 0
            success_rate = 100.0 if success else 0.0

            eval_rewards.append(total_reward)
            success_rates.append(success_rate)

            print(f"[Ep {episode}] Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Total Reward: {total_reward:.2f}, Success: {success_rate:.1f}%")

    env.close()
    return training_actor_losses, training_critic_losses, eval_rewards, success_rates
