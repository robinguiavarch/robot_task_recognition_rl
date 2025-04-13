import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    """
    A simple feedforward neural network for Q-learning.
    Input: state vector
    Output: Q-values for each possible action
    """
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)


def select_action(model, state, eps=0.1, device="cpu"):
    """
    Epsilon-greedy action selection.

    Args:
        model: The Q-network.
        state: Encoded input state (numpy array).
        eps: Exploration probability.
        device: Torch device.

    Returns:
        action (int): Selected action index (0 to 7).
    """
    if np.random.rand() < eps:
        return np.random.randint(0, 8)
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = model(state_tensor)
        return torch.argmax(q_values).item()


def train(q_net, target_net, buffer, optimizer, batch_size, gamma, device="cpu"):
    """
    Perform a single training step using a batch of transitions from the replay buffer.

    Args:
        q_net: Current Q-network.
        target_net: Target Q-network.
        buffer: ReplayBuffer object.
        optimizer: Optimizer for the Q-network.
        batch_size: Number of transitions to sample.
        gamma: Discount factor.
        device: Torch device.

    Returns:
        float or None: Loss value if a batch is sampled, else None.
    """
    if len(buffer) < batch_size:
        return None

    state, action, reward, next_state, done = buffer.sample(batch_size)

    state = torch.tensor(state, dtype=torch.float32).to(device)
    action = torch.tensor(action, dtype=torch.float32).to(device)
    reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(device)
    next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
    done = torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(device)

    q_values = q_net(state)
    target_q_values = target_net(next_state).detach()
    max_target_q = target_q_values.max(1, keepdim=True)[0]

    q_action = torch.sum(q_values * action, dim=1, keepdim=True)
    target = reward + (1 - done) * gamma * max_target_q

    loss = F.mse_loss(q_action, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
