"""
encoder_opta.py  –  shared encoders for the *On-Policy Transformer Actor* (“OPTA”)

* convert_index_to_action : int 0-7  →  3-bit binary vector
* encode_symbol_bg_fg     : event dict → 33-D observation vector
* action_encoder          : 3-bit vector **or** int 0-7 → 8-D one-hot

"""

from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
#                         1)  ACTION  INT  ↔  BINARY                          #
# --------------------------------------------------------------------------- #

def convert_index_to_action(index: int) -> np.ndarray:
    """
    Convert an integer index (0 to 7) into a 3-bit binary action vector.

    Example
    -------
    >>> convert_index_to_action(4)
    array([1, 0, 0])
    """
    assert 0 <= index < 8, "Index must be between 0 and 7 (inclusive)"
    return np.array(
        [int(bit) for bit in bin(index).removeprefix("0b").zfill(3)],
        dtype=np.int8,
    )


# --------------------------------------------------------------------------- #
#                       2)  OBSERVATION  (33-D)                               #
# --------------------------------------------------------------------------- #

def encode_symbol_bg_fg(
    event: dict,
    all_types: list[str],
    all_attributes: dict[str, list[str]],
) -> np.ndarray:
    """
    Encode a single event into a 33-dimensional feature vector:

        [ symbol_one_hot (15)  |  bg_one_hot (8)  |
          fg_one_hot (8)       |  start_time  |  end_time ]

    If a field is missing, the corresponding slice is left at zero.
    """
    symbol_vec = np.zeros(len(all_types), dtype=np.float32)
    bg_vec     = np.zeros(len(all_attributes["bg"]), dtype=np.float32)
    fg_vec     = np.zeros(len(all_attributes["fg"]), dtype=np.float32)

    # 1) symbol
    symbol = event.get("symbol")
    if symbol in all_types:
        symbol_vec[all_types.index(symbol)] = 1.0

    # 2) background colour
    bg = event.get("bg_color")
    if bg in all_attributes["bg"]:
        bg_vec[all_attributes["bg"].index(bg)] = 1.0

    # 3) foreground colour
    fg = event.get("symbol_color")
    if fg in all_attributes["fg"]:
        fg_vec[all_attributes["fg"].index(fg)] = 1.0

    # 4) timing (default 0.0 when absent)
    start = float(event.get("start_time", 0.0))
    end   = float(event.get("end_time",   0.0))

    return np.concatenate(
        [symbol_vec, bg_vec, fg_vec, np.array([start, end], dtype=np.float32)]
    )


# --------------------------------------------------------------------------- #
#               3)  ACTION  (binary 3)  →  ONE-HOT (8)                        #
# --------------------------------------------------------------------------- #

def action_encoder(action_or_index) -> np.ndarray:
    """
    Encode *either* a 3-bit binary vector **or** an integer in [0, 7]
    into a one-hot vector of size 8.

    Examples
    --------
    >>> action_encoder([1, 0, 0])
    array([0., 0., 0., 0., 1., 0., 0., 0.])

    >>> action_encoder(3)
    array([0., 0., 0., 1., 0., 0., 0., 0.])
    """
    if isinstance(action_or_index, (list, np.ndarray)):
        bits = np.asarray(action_or_index, dtype=int)
        assert bits.shape == (3,), "Binary action must have exactly 3 elements"
        index = bits[0] * 4 + bits[1] * 2 + bits[2] * 1
    elif isinstance(action_or_index, int):
        index = action_or_index
        assert 0 <= index < 8, "Action index must be in range [0, 7]"
    else:
        raise TypeError(
            "action_or_index must be a 3-bit list/array or an integer index"
        )

    one_hot = np.zeros(8, dtype=np.float32)
    one_hot[index] = 1.0
    return one_hot
