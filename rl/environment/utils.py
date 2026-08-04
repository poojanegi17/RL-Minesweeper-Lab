"""Small shared helpers used by the environment and agents."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def action_to_coords(action: int, cols: int) -> Tuple[int, int]:
    """Convert a flattened action index into (row, col) board coordinates."""
    row, col = divmod(action, cols)
    return row, col


def coords_to_action(row: int, col: int, cols: int) -> int:
    """Convert (row, col) board coordinates into a flattened action index."""
    return row * cols + col


def board_to_array(board: List[List[int]]) -> np.ndarray:
    """Convert the engine's nested-list board into a numpy array."""
    return np.array(board, dtype=np.int8)
