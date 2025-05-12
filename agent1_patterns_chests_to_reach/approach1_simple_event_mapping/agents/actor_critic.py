import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """
    π_θ(a|s): maps state → action probabilities
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state: (batch, state_dim)
        # returns: (batch, action_dim) probabilities
        return self.net(state)


class CriticQ(nn.Module):
    """
    Q_φ(s,a): maps state → a Q-value for each action
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state: (batch, state_dim)
        # returns: (batch, action_dim) Q-values
        return self.net(state)


def select_action(actor, state, device="cpu"):
    if isinstance(state, np.ndarray):
        state = torch.tensor(state, dtype=torch.float32)
    state = state.unsqueeze(0).to(device)  # (1, state_dim)
    
    probs = actor(state)                  # (1, action_dim)
    dist  = Categorical(probs.squeeze(0)) # (action_dim,)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


def compute_targets(
    rewards: list[float],
    next_states: list[torch.Tensor],
    dones: list[bool],
    critic: CriticQ,
    gamma: float
) -> torch.Tensor:
    """
    Compute target Q-values y_t = r_t + γ * max_a' Q(s_{t+1},a') * (1 - done).
    Returns a tensor of shape (T,).
    """
    ys = []
    with torch.no_grad():
        for r, s_next, done in zip(rewards, next_states, dones):
            s1 = s_next.unsqueeze(0).to(device)       # (1,state_dim)
            q1 = critic(s1)                           # (1,action_dim)
            max_q1 = q1.max(dim=-1)[0]                # (1,)
            y = r + gamma * max_q1 * (1.0 - float(done))
            ys.append(y.squeeze(0))
    return torch.stack(ys)                           # (T,)


def update(
    actor: Actor,
    critic: CriticQ,
    optimizer_actor: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer,
    states: list[torch.Tensor],
    actions: list[int],
    log_probs: list[torch.Tensor],
    rewards: list[float],
    next_states: list[torch.Tensor],
    dones: list[bool],
    gamma: float
) -> (float, float):
    """
    Perform one joint update of the CriticQ and the Actor:

    1) Critic loss: MSE between Q(s,a) and target y.
    2) Actor loss: REINFORCE with advantage A_t = y - Q(s,a).
    """

    # stack into tensors
    S  = torch.stack(states).to(device)                    # (T,state_dim)
    A  = torch.tensor(actions, dtype=torch.int64).to(device)  # (T,)
    LP = torch.stack(log_probs).to(device)                 # (T,)

    # 1) Critic update
    Qs   = critic(S)                                       # (T,action_dim)
    Q_sa = Qs.gather(1, A.unsqueeze(1)).squeeze(1)         # (T,)

    y     = compute_targets(rewards, next_states, dones, critic, gamma)  # (T,)
    critic_loss = F.mse_loss(Q_sa, y)

    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step()

    # 2) Actor update
    with torch.no_grad():
        advantage = y - Q_sa                               # (T,)

    actor_loss = -(LP * advantage).mean()

    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()

    return actor_loss.item(), critic_loss.item()
