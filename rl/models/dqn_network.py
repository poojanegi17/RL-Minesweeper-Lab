"""CNN-based Q-network for Minesweeper board states."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
from torch import nn

# Channel 0: hidden-cell mask.
# Channel 1: revealed-cell mask.
# Channels 2-10: one-hot encoding of the revealed neighbor-mine count (0-8).
NUM_CHANNELS = 11

# Named architecture configs, passed as `**preset` into `DQNNetwork`. "default"
# matches the network this project has used since the CNN was first introduced;
# "small" has roughly a quarter of the conv capacity and half the linear width,
# for testing whether reduced capacity is inherently more stable over long
# training runs (see the DQN Experiments section in the README).
NETWORK_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {"conv_channels": (16, 32), "hidden_dim": 128},
    "small": {"conv_channels": (8, 16), "hidden_dim": 64},
}


def encode_observation(observation: Sequence[Sequence[int]]) -> np.ndarray:
    """Encode a raw board observation into semantic channels for the network.

    A single normalized scalar per cell forces the network to infer, purely
    from magnitude, that "hidden" and "revealed" are different *kinds* of
    information rather than points on a continuum -- and -1 (hidden) sits
    numerically right next to 0 (revealed, zero neighboring mines), the two
    states that matter most to tell apart. One-hot channels make that
    distinction explicit and structural instead: hidden cells activate only
    channel 0, and a revealed cell showing count `v` activates channel 1
    (revealed) and channel `2 + v` (its exact count), so a Conv2d kernel can
    read "which category" directly from which channels are lit at a given
    board position, without needing to first learn to disentangle it.

    Args:
        observation: A (rows, cols) board with -1 for hidden cells and 0-8
            for revealed cells.

    Returns:
        A float32 array of shape (NUM_CHANNELS, rows, cols).
    """
    board = np.asarray(observation, dtype=np.int64)
    rows, cols = board.shape
    channels = np.zeros((NUM_CHANNELS, rows, cols), dtype=np.float32)

    channels[0] = (board == -1).astype(np.float32)
    channels[1] = (board != -1).astype(np.float32)
    for value in range(9):
        channels[2 + value] = (board == value).astype(np.float32)

    return channels


class DQNNetwork(nn.Module):
    """A small CNN mapping a multi-channel board state to one Q-value per cell.

    Architecture (sizes configurable; shown here for the "default" preset):
        Observation (NUM_CHANNELS, rows, cols)
            -> Conv2d(NUM_CHANNELS, 16, 3x3) + ReLU
            -> Conv2d(16, 32, 3x3) + ReLU
            -> Flatten
            -> Linear(32*rows*cols, 128) + ReLU
            -> Linear(128, rows*cols)   # raw Q-values, one per cell

    Convolutions are used instead of a plain MLP because Minesweeper's number
    clues are inherently spatial: what a cell means depends on its local
    neighborhood, which is exactly the pattern convolutional kernels learn to
    detect regardless of where on the board it occurs. `padding=1` keeps the
    spatial size fixed at every conv layer, so the flattened dimension only
    depends on `rows * cols` (and the second conv-layer width).
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        in_channels: int = NUM_CHANNELS,
        conv_channels: Tuple[int, int] = (16, 32),
        hidden_dim: int = 128,
    ) -> None:
        """Build the network for a board of the given size.

        Args:
            rows: Number of board rows.
            cols: Number of board columns.
            in_channels: Number of input channels; must match the channel
                count produced by `encode_observation`.
            conv_channels: `(first_layer_filters, second_layer_filters)`.
                Pass a preset from `NETWORK_PRESETS` rather than hand-tuning
                these directly, unless running a new architecture experiment.
            hidden_dim: Width of the fully-connected layer between the
                flattened conv output and the Q-value output layer.
        """
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.hidden_dim = hidden_dim
        self.num_actions = rows * cols

        c1, c2 = conv_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * rows * cols, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of board states to Q-values.

        Args:
            x: Tensor of shape (batch_size, in_channels, rows, cols).

        Returns:
            Tensor of shape (batch_size, rows * cols) with one raw Q-value
            (no output activation) per cell.
        """
        return self.head(self.conv(x))
