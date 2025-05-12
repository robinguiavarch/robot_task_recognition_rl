import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    """
    A feedforward neural network for Q-learning.
    - Input: State vector (e.g., 16D if using a sliding window of 4 events).
    - Output: Q-values for each possible action (8 in this example).

    Args:
        state_size (int): Dimensionality of the input state.
        action_size (int): Number of possible actions (8).
    """
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        """
        Forward pass through the Q-network.

        Args:
            x (torch.Tensor): Input state tensor of shape (batch_size, state_size).

        Returns:
            torch.Tensor: Q-values of shape (batch_size, action_size).
        """
        return self.fc(x)


def select_action(model, state, eps=0.1, device="cpu"):
    """
    Epsilon-greedy action selection.

    Args:
        model (QNetwork): The Q-network.
        state (np.ndarray): Encoded input state (e.g., shape (16,)).
        eps (float): Exploration probability.
        device (str): Torch device ("cpu" or "cuda").

    Returns:
        int: Chosen action index in [0..7].
    """
    if np.random.rand() < eps:
        return np.random.randint(0, 8)
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = model(state_tensor)
        return torch.argmax(q_values, dim=1).item()


def train(q_net, target_net, buffer, optimizer, batch_size, gamma, device="cpu"):
    """
    Perform one training step using a batch of transitions from the replay buffer.

    Args:
        q_net (QNetwork): The main Q-network to train.
        target_net (QNetwork): The target Q-network for stable Q-learning updates.
        buffer (ReplayBuffer): ReplayBuffer instance to sample transitions from.
        optimizer (torch.optim.Optimizer): Optimizer for q_net.
        batch_size (int): Number of transitions per mini-batch.
        gamma (float): Discount factor.
        device (str): Torch device ("cpu" or "cuda").

    Returns:
        float or None: The loss value if training occurs, otherwise None.
    """
    # If the buffer doesn't have enough samples yet, do nothing.
    if len(buffer) < batch_size:
        return None

    # Sample a batch of transitions: state, action, reward, next_state, done
    state, action, reward, next_state, done = buffer.sample(batch_size)

    # Convert to tensors and send to device
    state = torch.tensor(state, dtype=torch.float32).to(device)
    action = torch.tensor(action, dtype=torch.float32).to(device)
    reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(device)
    next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
    done = torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(device)

    # Forward pass through the current Q-network
    q_values = q_net(state)
    # Forward pass through the target network (no gradient)
    target_q_values = target_net(next_state).detach()
    max_target_q = target_q_values.max(dim=1, keepdim=True)[0]

    # Extract Q-values for the taken actions
    # (each action is one-hot, so sum(q_values * action, dim=1) is Q(s,a))
    q_action = torch.sum(q_values * action, dim=1, keepdim=True)

    # Compute the target: r + gamma * max(Q_target) * (1 - done)
    target = reward + (1 - done) * gamma * max_target_q

    # Compute the loss (MSE between current Q and target)
    loss = F.mse_loss(q_action, target)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
