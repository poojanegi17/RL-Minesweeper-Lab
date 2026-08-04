"""Fixed-capacity experience replay buffer for off-policy training."""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    """A ring buffer of (state, action, reward, next_state, done) transitions.

    Domain-agnostic: it has no knowledge of Minesweeper or of the network's
    expected input format, it simply stores whatever arrays it is given and
    samples them uniformly at random. Random sampling breaks the temporal
    correlation between consecutive transitions from the same episode, which
    is what lets a single online network be trained with ordinary
    i.i.d.-style minibatch gradient descent.
    """

    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        """Create a replay buffer.

        Args:
            capacity: Maximum number of transitions retained. Once full, the
                oldest transition is evicted for every new one pushed.
            seed: Optional seed for reproducible sampling.
        """
        self.capacity = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add one transition, evicting the oldest transition if at capacity."""
        self._buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly sample `batch_size` transitions, stacked into batched arrays.

        Returns:
            A tuple `(states, actions, rewards, next_states, dones)` where
            `states`/`next_states` have shape `(batch_size, *state_shape)` and
            the rest have shape `(batch_size,)`.
        """
        batch = self._rng.sample(list(self._buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)
