"""Actor-Critic CNN for PPO on Minesweeper board states.

Shares `models.dqn_network.encode_observation`'s 11-channel one-hot input
(the same categorical-encoding argument from that module applies here
unchanged: Minesweeper's clues are categorical, not continuous, so making
"hidden vs. revealed-with-count-N" explicit as separate channels is a better
fit than a single normalized scalar) and a similar convolutional trunk to
`DQNNetwork`. The difference is what sits on top of the trunk: DQN has one
head producing a Q-value per cell; PPO needs two heads sharing the same
feature extractor -- an actor producing one action logit per cell, and a
critic producing a single state-value estimate -- since policy-gradient
methods need both "what should I do" and "how good is this state" as
separate quantities to compute an advantage.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn

from models.dqn_network import NUM_CHANNELS

# Mirrors `dqn_network.NETWORK_PRESETS`, deliberately using the same names and
# the same shapes so a PPO/DQN comparison at one preset is a comparison of the
# algorithms rather than of two different networks.
#
# PPO ran on "default" for this project's entire history, which is why every
# published PPO result predates the head change: `--network-size` did not exist
# and there was no way to ask for anything else. That matters most for the
# reason given in point 3 of `dqn_network`'s own preset comment -- a Linear head
# is board-size-specific, so a PPO model trained at 5x5 cannot be loaded at 9x9
# and each level had to train from scratch, which is exactly the constraint
# `fully_conv` removed for DQN.
NETWORK_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {"conv_channels": (16, 32), "hidden_dim": 128},
    "small": {"conv_channels": (8, 16), "hidden_dim": 64},
    "deep": {"conv_channels": (16, 32, 32, 32), "hidden_dim": 128},
    "fully_conv": {"conv_channels": (16, 32, 32, 32), "hidden_dim": 128, "head_type": "conv"},
}

# "linear" is the original design and stays the default so every PPO checkpoint
# this project has already published still loads. See `DQNNetwork` for the
# argument behind "conv"; the only PPO-specific part is the critic, described
# where it is built below.
HEAD_TYPES = ("linear", "conv")


class PPONetwork(nn.Module):
    """Shared CNN trunk with separate actor (policy) and critic (value) heads.

    Architecture (sizes configurable; shown here for the defaults):
        Observation (NUM_CHANNELS, rows, cols)
            -> Conv2d(NUM_CHANNELS, 16, 3x3) + ReLU
            -> Conv2d(16, 32, 3x3) + ReLU
            -> Flatten
            -> Linear(32*rows*cols, 128) + ReLU      # shared trunk
            -> Linear(128, rows*cols)                # actor: raw action logits
            -> Linear(128, 1)                        # critic: state value

    The actor's output is raw logits over *all* cells, unmasked -- action
    masking to only-hidden cells (mirroring every other agent in this
    project) is the caller's responsibility, done with the observation's own
    hidden-mask channel, not baked into the network itself.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        in_channels: int = NUM_CHANNELS,
        conv_channels: Tuple[int, ...] = (16, 32),
        hidden_dim: int = 128,
        head_type: str = "linear",
    ) -> None:
        """Build the network for a board of the given size.

        Args:
            rows: Number of board rows.
            cols: Number of board columns.
            in_channels: Number of input channels; must match the channel
                count produced by `encode_observation`.
            conv_channels: `(first_layer_filters, second_layer_filters)`.
            hidden_dim: Width of the shared fully-connected trunk feeding
                both heads.
        """
        super().__init__()
        if head_type not in HEAD_TYPES:
            raise ValueError(f"Unknown head_type {head_type!r}; choose from {HEAD_TYPES}")
        if len(conv_channels) == 0:
            raise ValueError("conv_channels must name at least one layer")

        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.hidden_dim = hidden_dim
        self.head_type = head_type
        self.num_actions = rows * cols

        layers: list[nn.Module] = []
        channels_in = in_channels
        for channels_out in conv_channels:
            layers.append(nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            channels_in = channels_out
        self.conv = nn.Sequential(*layers)

        if head_type == "conv":
            # The conv stack is itself the shared trunk here -- there is no
            # flatten-and-Linear stage to share, so `trunk` is a no-op kept for
            # a single `forward` path.
            self.trunk = nn.Identity()
            # Actor: a per-cell MLP via 1x1 convolutions, identical in form to
            # `DQNNetwork`'s conv head -- one logit per board position, the same
            # weights applied everywhere.
            self.actor_head = nn.Sequential(
                nn.Conv2d(conv_channels[-1], hidden_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
                nn.Flatten(),
            )
            # Critic: the same per-cell reduction, then a mean over positions.
            # A value head has to emit one scalar per state, and the obvious
            # board-size-independent way to get there is to score every cell and
            # average -- global pooling rather than a Linear over a flattened
            # map, which would reintroduce exactly the rows*cols dependence this
            # head exists to remove. `nn.AdaptiveAvgPool2d(1)` averages over
            # whatever spatial extent it is given, so the critic transfers
            # across board sizes on the same terms as the actor.
            self.critic_head = nn.Sequential(
                nn.Conv2d(conv_channels[-1], hidden_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
        else:
            self.trunk = nn.Sequential(
                nn.Flatten(),
                nn.Linear(conv_channels[-1] * rows * cols, hidden_dim),
                nn.ReLU(),
            )
            self.actor_head = nn.Linear(hidden_dim, self.num_actions)
            self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map a batch of board states to (action logits, state values).

        Args:
            x: Tensor of shape (batch_size, in_channels, rows, cols).

        Returns:
            `(action_logits, values)`: `action_logits` has shape
            `(batch_size, rows * cols)` (raw, unmasked, no output
            activation); `values` has shape `(batch_size, 1)`.
        """
        features = self.trunk(self.conv(x))
        return self.actor_head(features), self.critic_head(features)
