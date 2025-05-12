import numpy as np
from collections import deque

# Size of the sliding window
WINDOW_SIZE = 4
# Internal history: keeps the last encoded events
_event_history = deque(maxlen=WINDOW_SIZE)

def encode_symbol_bg_fg(event: dict, all_types, all_attributes) -> np.ndarray:
    """
    Encode an event into a one-hot vector plus timing:
    [symbol_oh (15D), bg_oh (8D), fg_oh (8D), start_time, end_time].

    Returns:
        np.ndarray: A 33-dimensional encoded feature vector.
    """
    symbol_vec = np.zeros(len(all_types), dtype=np.float32)
    bg_vec = np.zeros(len(all_attributes["bg"]), dtype=np.float32)
    fg_vec = np.zeros(len(all_attributes["fg"]), dtype=np.float32)

    # 1) Encode the symbol
    symbol = event.get("symbol", None)
    if symbol in all_types:
        symbol_idx = all_types.index(symbol)
        symbol_vec[symbol_idx] = 1.0

    # 2) Encode background color
    bg_color = event.get("bg_color", None)
    if bg_color in all_attributes["bg"]:
        bg_idx = all_attributes["bg"].index(bg_color)
        bg_vec[bg_idx] = 1.0

    # 3) Encode foreground color
    fg_color = event.get("symbol_color", None)
    if fg_color in all_attributes["fg"]:
        fg_idx = all_attributes["fg"].index(fg_color)
        fg_vec[fg_idx] = 1.0

    # 4) Encode timing information
    # If the event has no time field, default to 0.0
    start_time = float(event.get("start_time", 0.0))
    end_time   = float(event.get("end_time", 0.0))

    # Final concatenation: 15 + 8 + 8 + 2 = 33D
    return np.concatenate([
        symbol_vec, 
        bg_vec, 
        fg_vec, 
        np.array([start_time, end_time], dtype=np.float32)
    ], axis=0)

def sliding_window_encoder(event: dict, all_types, all_attributes) -> np.ndarray:
    """
    Encode an event using a sliding window of past events.

    A single event is 33D → final output is 132D with window size 4.

    Returns:
        np.ndarray: Flattened vector of all events in the current window.
    """
    encoded_single = encode_symbol_bg_fg(event, all_types, all_attributes)
    _event_history.append(encoded_single)

    history_list = list(_event_history)

    while len(history_list) < WINDOW_SIZE:
        history_list.insert(0, np.zeros_like(encoded_single))  # pad with 33D zero vectors

    final_encoding = np.concatenate(history_list, axis=0)
    return final_encoding

def reset_sliding_window():
    """
    Clear the internal sliding window history.
    Should be called at the beginning of each episode.
    """
    _event_history.clear()

def action_encoder(action_or_index) -> np.ndarray:
    """
    Encode either a binary vector of size 3 or an integer (0–7)
    into a one-hot encoded vector of size 8.

    Args:
        action_or_index (np.ndarray, list, or int): Binary action like [1, 0, 0] or integer index (0–7)

    Returns:
        np.ndarray: One-hot encoded vector (size 8)

    Examples:
        [1, 0, 0] → index 4 → one-hot [0, 0, 0, 0, 1, 0, 0, 0]
        3 → one-hot [0, 0, 0, 1, 0, 0, 0, 0]
    """
    if isinstance(action_or_index, (np.ndarray, list)):
        action_vec = np.array(action_or_index, dtype=int)
        assert action_vec.shape == (3,), "Binary action must contain exactly 3 elements"
        index = int(action_vec[0] * 4 + action_vec[1] * 2 + action_vec[2] * 1)
    elif isinstance(action_or_index, int):
        index = action_or_index
        assert 0 <= index < 8, "Action index must be in the range [0, 7]"
    else:
        raise TypeError("action_or_index must be a binary vector (list or np.ndarray) or an integer")

    one_hot = np.zeros(8)
    one_hot[index] = 1
    # print(f"[action_encoder] Input {action_or_index} → Index {index} → One-hot {one_hot}")
    return one_hot

def convert_index_to_action(index: int) -> np.ndarray:
    """
    Convert an integer index (0 to 7) into a 3-bit binary action vector.

    Example:
        4 → [1, 0, 0]

    Returns:
        np.ndarray: Binary vector representing the action.
    """
    assert 0 <= index < 8, "Index must be between 0 and 7 (inclusive)"
    binary_action = np.array([int(digit) for digit in bin(index).removeprefix("0b").zfill(3)])
    # print(f"[convert_index_to_action] Index {index} -> Binary {binary_action}")
    return binary_action
