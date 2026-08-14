"""CNN-based Q-network for Minesweeper board states."""

from __future__ import annotations

from typing import Any, Dict, Sequence

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
    # Four conv layers at the same widths the default's two use, so *depth* is
    # the only thing that changes. The motivation is specific: Minesweeper's
    # deduction is iterative -- a revealed 0 makes its neighbours safe in one
    # round, the subset rule needs two constraints compared, and a chained
    # deduction (prove a mine, then use it to prove a safe cell) needs three or
    # more. A conv stack is a fixed-depth circuit, so two layers can implement
    # about two rounds of that inference; `agents/csp_solver.py` instead
    # iterates its rules to a fixpoint. Measured against a trained agent, a
    # provably-safe cell was available on 37.7% of its moves and it picked one
    # only 62.5% of the time -- "gets the easy deductions, misses the chained
    # ones", which is what motivated adding layers rather than width.
    "deep": {"conv_channels": (16, 32, 32, 32), "hidden_dim": 128},
    # Same depth as "deep", but the flatten-and-Linear head is replaced by a
    # position-wise 1x1 convolution head (`head_type="conv"`). Three things
    # follow from that, and they're the reason this is the recommended preset
    # for new work:
    #
    #   1. Parameter count collapses. The Linear head is where nearly all the
    #      weights live -- on 5x5 it is 94% of the "default" network, and on
    #      16x16 it is 99.4% (a single Linear(8192, 128) is over a million
    #      parameters). A 1x1 conv head is a few thousand regardless of board
    #      size, which matters a great deal at this project's step budget.
    #   2. Translation equivariance. "A revealed 1 with exactly one hidden
    #      neighbour means that neighbour is a mine" is the same rule at every
    #      board position. A Linear head reads a flattened vector, so it has to
    #      relearn that rule separately for each of the rows*cols positions; a
    #      conv head learns it once and applies it everywhere.
    #   3. Board-size independence. Nothing in this architecture depends on
    #      rows/cols, so one set of weights runs on any board. That makes it
    #      possible to train on a small board -- where episodes are cheap -- and
    #      evaluate zero-shot on 9x9 or 16x16, which is otherwise out of reach
    #      at ~100 environment steps/second.
    "fully_conv": {"conv_channels": (16, 32, 32, 32), "hidden_dim": 128, "head_type": "conv"},
}

# Head architectures. "linear" flattens the conv stack and runs it through a
# fully-connected layer -- the original design, kept as the default so every
# checkpoint this project has already published still loads. "conv" replaces
# that with 1x1 convolutions; see the "fully_conv" preset above.
HEAD_TYPES = ("linear", "conv")


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

    Architecture (depth and widths both configurable via `conv_channels`;
    shown here for the "default" preset):
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
        conv_channels: Sequence[int] = (16, 32),
        hidden_dim: int = 128,
        head_type: str = "linear",
    ) -> None:
        """Build the network for a board of the given size.

        Args:
            rows: Number of board rows.
            cols: Number of board columns.
            in_channels: Number of input channels; must match the channel
                count produced by `encode_observation`.
            conv_channels: One filter count per convolutional layer, so its
                length sets the network's depth. Pass a preset from
                `NETWORK_PRESETS` rather than hand-tuning these directly,
                unless running a new architecture experiment.
            hidden_dim: Width of the layer between the conv stack and the
                Q-value output. For `head_type="linear"` this is a
                fully-connected layer over the flattened conv output; for
                `head_type="conv"` it is the channel count of an intermediate
                1x1 convolution, which is the same computation applied
                independently at every board position.
            head_type: One of `HEAD_TYPES`. "conv" makes the network
                board-size independent -- `rows` and `cols` then only set
                `num_actions` for bookkeeping and place no constraint on the
                inputs `forward` accepts.

        Raises:
            ValueError: If `conv_channels` is empty or `head_type` is unknown.
        """
        super().__init__()
        if head_type not in HEAD_TYPES:
            raise ValueError(f"Unknown head_type {head_type!r}; choose from {HEAD_TYPES}")

        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.hidden_dim = hidden_dim
        self.head_type = head_type
        self.num_actions = rows * cols

        if len(conv_channels) == 0:
            raise ValueError("conv_channels must name at least one layer")
        layers: list[nn.Module] = []
        channels_in = in_channels
        for channels_out in conv_channels:
            layers.append(nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            channels_in = channels_out
        self.conv = nn.Sequential(*layers)

        if head_type == "conv":
            # Two 1x1 convolutions are a per-cell MLP: each board position is
            # mapped from conv_channels[-1] features to a single Q-value using
            # the *same* weights everywhere. Flatten then puts those per-cell
            # values into the (batch, rows*cols) layout the agent's action
            # indexing expects, matching the linear head's output contract.
            self.head = nn.Sequential(
                nn.Conv2d(conv_channels[-1], hidden_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
                nn.Flatten(),
            )
        else:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(conv_channels[-1] * rows * cols, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_actions),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of board states to Q-values.

        Args:
            x: Tensor of shape (batch_size, in_channels, height, width). With
                `head_type="conv"`, height and width are free -- they need not
                be the `rows`/`cols` the network was constructed with, which is
                what allows a small-board-trained network to be evaluated on a
                larger board. With `head_type="linear"` they must match, or the
                flattened dimension won't line up with the Linear layer.

        Returns:
            Tensor of shape (batch_size, height * width) with one raw Q-value
            (no output activation) per cell, in row-major order to match
            `environment.utils.coords_to_action`.
        """
        return self.head(self.conv(x))
