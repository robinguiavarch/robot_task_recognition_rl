import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)  # Output a probability distribution over actions
        )

    def forward(self, state):
        return self.policy_net(state)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.value_net(state)


def select_action(actor, state, device=device):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    probs = actor(state_tensor)
    dist = Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


def compute_returns(rewards, gamma):
    """
    Compute cumulative discounted returns for a list of rewards.
    """
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def update(actor, critic, optimizer_actor, optimizer_critic, states, actions, log_probs, rewards, gamma, device=device):
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    log_probs = torch.stack(log_probs).to(device)
    returns = compute_returns(rewards, gamma)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    # Critic loss: mean squared error between predicted value and return
    values = critic(states).squeeze()
    critic_loss = F.mse_loss(values, returns)

    # Advantage: return - value
    advantages = returns - values.detach()

    # Actor loss: negative log prob * advantage
    actor_loss = -(log_probs * advantages).mean()

    # Update both networks
    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()

    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step()

    return actor_loss.item(), critic_loss.item()
