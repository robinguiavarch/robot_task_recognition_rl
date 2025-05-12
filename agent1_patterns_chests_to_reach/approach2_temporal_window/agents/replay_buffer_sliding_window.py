# replay_buffer.py

import random
from collections import deque
import numpy as np

class ReplayBuffer:
    """
    A simple replay buffer for experience replay in Reinforcement Learning.

    This implementation stores transitions of the form:
    (state, action, reward, next_state, done),
    and can return random mini-batches of these transitions.

    Attributes:
        capacity (int): Maximum number of transitions to store.
        buffer (collections.deque): Internal deque storing the transitions.
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
        Store a single transition in the replay buffer.

        Args:
            state (np.ndarray): Encoded current state (e.g., 16D vector if using a sliding window).
            action (np.ndarray or int): Action taken (e.g., one-hot 8D or an int in [0..7]).
            reward (float): Reward received from the environment.
            next_state (np.ndarray): Encoded next state.
            done (bool): Whether the episode ended after this transition.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a random batch of transitions from the replay buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple of np.ndarray: (state, action, reward, next_state, done)
                Each returned array has shape (batch_size, ...),
                depending on the dimensionality of the stored data.
        """
        batch = random.sample(self.buffer, batch_size)
        # Unzip the list of tuples into separate arrays
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        """
        Returns:
            int: The current number of transitions stored.
        """
        return len(self.buffer)
