# replay_buffer.py

import random
from collections import deque
import numpy as np

class ReplayBuffer:
    """
    A simple replay buffer to store transitions for experience replay.

    Attributes:
        capacity (int): Maximum number of transitions to store.
        buffer (collections.deque): Internal buffer storing the transitions.
    """
    def __init__(self, capacity):
        """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum number of transitions to keep in the buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        Args:
            state (np.array): Current state.
            action (np.array or int): Action taken.
            reward (float): Reward received.
            next_state (np.array): Next state after the action.
            done (bool): Whether the episode ended after this transition.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple of arrays (state, action, reward, next_state, done).
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        """
        Return the current size of the replay buffer.
        """
        return len(self.buffer)
