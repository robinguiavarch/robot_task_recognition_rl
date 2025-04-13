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
    action_encoder,
    convert_index_to_action,
)
from agent1_patterns_chests_to_reach.utils.event_encoding import (
    event_to_dict_from_gym as event_from_obs_gym,
)
from agent1_patterns_chests_to_reach.approach1_simple_event_mapping.agents.replay_buffer import ReplayBuffer
from agent1_patterns_chests_to_reach.approach1_simple_event_mapping.evaluation.evaluation_dqn import evaluate_dqn
from agent1_patterns_chests_to_reach.approach1_simple_event_mapping.agents.dqn_agent import (
    QNetwork,
    select_action,
    train,
)

def train_agent(
    env_name="OpenTheChests-v0",
    buffer_capacity=1000,
    epochs=500,
    batch_size=128,
    learning_rate=0.01,
    gamma=0.99,
    evaluation_interval=50,
    evaluation_episodes=100,
    eps=0.1,
):
    register_custom_envs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_size = 4
    action_size = 8

    env = gym.make(env_name)
    q_net = QNetwork(state_size, action_size).to(device)
    target_q_net = QNetwork(state_size, action_size).to(device)
    target_q_net.load_state_dict(q_net.state_dict())

    optimizer = Adam(q_net.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(buffer_capacity)

    obs = env.reset()
    state = simple_encoder(event_from_obs_gym(obs, all_types, all_attributes))
    done = False
    training_loss = []
    average_rewards = []
    average_successes = []

    for episode in range(epochs):
        if done:
            obs = env.reset()
            state = simple_encoder(event_from_obs_gym(obs, all_types, all_attributes))
            done = False

        index_action = select_action(q_net, state, eps=eps, device=device)
        action = convert_index_to_action(index_action)

        obs, reward, done, info = env.step(action)
        next_state = simple_encoder(event_from_obs_gym(obs, all_types, all_attributes))

        buffer.push(state, action_encoder(index_action), reward, next_state, done)
        state = next_state

        loss = train(q_net, target_q_net, buffer, optimizer, batch_size=batch_size, gamma=gamma)
        if loss is not None:
            training_loss.append(loss)

        if episode % evaluation_interval == 0:
            target_q_net.load_state_dict(q_net.state_dict())
            test_env = gym.make(env_name)
            avg_reward, success_rate = evaluate_dqn(
                test_env, q_net, episodes=evaluation_episodes, device=device
            )
            average_rewards.append(avg_reward)
            average_successes.append(success_rate)
            print(
                f"[Gamma {gamma}] Episode {episode} - AvgReward: {avg_reward:.2f}, Success: {success_rate:.1f}%"
            )

    return training_loss, average_rewards, average_successes
