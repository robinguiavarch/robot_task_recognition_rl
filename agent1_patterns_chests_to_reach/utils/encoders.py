# agent1_patterns_chests_to_reach/wrappers/encoders.py

import numpy as np


def convert_index_to_action(index: int) -> np.ndarray:
    """
    Convert an integer index (0 to 7) into a 3-bit binary action vector.

    Example:
        4 → [1, 0, 0]
    """
    assert 0 <= index < 8, "Index must be between 0 and 7 (inclusive)"
    binary_action = np.array([int(digit) for digit in bin(index).removeprefix("0b").zfill(3)])
    # print(f"[convert_index_to_action] Index {index} -> Binary {binary_action}")
    return binary_action


def simple_encoder(state: dict) -> np.ndarray:
    """
    Encode a symbolic state (like {'symbol': 'B'}) into a one-hot vector of size 4.

    Mapping:
        - 'A' → index 1
        - 'B' → index 2
        - 'C' → index 3
        - unknown symbol or None → index 0

    Example:
        {'symbol': 'B'} → [0, 0, 1, 0]
    """
    symbol_to_index = {'A': 1, 'B': 2, 'C': 3}
    encoded = np.zeros(4)
    idx = symbol_to_index.get(state.get('symbol'), 0)
    encoded[idx] = 1
    # print(f"[simple_encoder] State {state} -> One-hot {encoded}")
    return encoded


def action_encoder(action_or_index) -> np.ndarray:
    """
    Encode either a binary vector of size 3 or an integer (0-7) into a one-hot vector of size 8.

    Args:
        action_or_index (np.ndarray, list, or int): Either a binary vector [1,0,0] or an index (0–7)

    Returns:
        np.ndarray: One-hot encoded vector of size 8

    Examples:
        [1, 0, 0] → index 4 → one-hot [0, 0, 0, 0, 1, 0, 0, 0]
        3 → one-hot [0, 0, 0, 1, 0, 0, 0, 0]
    """
    if isinstance(action_or_index, (np.ndarray, list)):
        action_vec = np.array(action_or_index, dtype=int)
        assert action_vec.shape == (3,), "Binary action must have exactly 3 elements"
        index = int(action_vec[0] * 4 + action_vec[1] * 2 + action_vec[2] * 1)
    elif isinstance(action_or_index, int):
        index = action_or_index
        assert 0 <= index < 8, "Action index must be in range [0, 7]"
    else:
        raise TypeError("action_or_index must be a binary vector (list or np.ndarray) or an integer")

    one_hot = np.zeros(8)
    one_hot[index] = 1
    # print(f"[action_encoder] Input {action_or_index} → Index {index} → One-hot {one_hot}")
    return one_hot
