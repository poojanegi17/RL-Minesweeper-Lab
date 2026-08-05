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

from typing import Tuple

import torch
from torch import nn

from models.dqn_network import NUM_CHANNELS


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
            hidden_dim: Width of the shared fully-connected trunk feeding
                both heads.
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
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * rows * cols, hidden_dim),
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
