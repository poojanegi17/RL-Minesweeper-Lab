"""Fixed-length on-policy rollout buffer for PPO.

Unlike `ReplayBuffer` (a fixed-capacity ring buffer sampled uniformly at
random for off-policy training), this buffer holds exactly one rollout's
worth of *sequential* transitions collected under the current policy, and is
discarded after the update that consumes it -- PPO's clipped objective is
only valid for data collected under the policy being updated, so nothing
here is ever replayed across updates the way DQN's transitions are.

Domain-agnostic like `ReplayBuffer`: it stores whatever arrays it is given
and has no knowledge of Minesweeper.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

Observation = np.ndarray


class RolloutBuffer:
    """Stores one rollout's transitions and computes GAE advantages/returns.

    Usage: call `add()` once per environment step during rollout collection,
    then `compute_gae()` once at the end of the rollout (after the final
    step's bootstrap value is known), then `get()` to retrieve stacked
    arrays for the PPO update. Call `reset()` before starting the next
    rollout.
    """

    def __init__(self) -> None:
        self.observations: List[Observation] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Clear all stored transitions and computed advantages/returns."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None

    def add(
        self,
        observation: Observation,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Append one environment step to the current rollout.

        Args:
            observation: The *encoded* state the action was chosen from
                (e.g. `models.dqn_network.encode_observation`'s output), not
                the raw board -- stored pre-encoded so `get()` can hand back
                a network-ready batch directly.
            action: The discrete action taken.
            reward: The reward received for taking `action`.
            done: Whether this step ended the episode (terminated or
                truncated), used by `compute_gae` to avoid bootstrapping
                across episode boundaries.
            log_prob: log pi(action | observation) under the policy that
                chose it, needed later for PPO's probability ratio.
            value: V(observation) under the critic at collection time.
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        """Compute Generalized Advantage Estimation advantages and returns.

        Standard recursive GAE, walking the rollout backwards:
            delta_t   = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t       = delta_t + gamma * gae_lambda * (1 - done_t) * A_{t+1}
            return_t  = A_t + V(s_t)

        `V(s_{t+1})` is the next stored value for every step except the
        last, where it is `last_value` -- the critic's estimate of the state
        immediately following the rollout (0.0 if that state is terminal),
        supplied by the caller since it isn't part of this rollout's stored
        transitions. `(1 - done_t)` zeroes out bootstrapping across episode
        boundaries in either term, so multiple episodes within one rollout
        are handled correctly without special-casing.

        Populates `self.advantages` and `self.returns` (each shape `(T,)`,
        `T = len(self)`).
        """
        length = len(self)
        advantages = np.zeros(length, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(length)):
            next_value = values[t + 1] if t + 1 < length else last_value
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + values

    def get(self) -> Dict[str, np.ndarray]:
        """Return the rollout as stacked arrays, ready for a PPO update.

        Must be called after `compute_gae`. Returns a dict with
        `observations` (shape `(T, *obs_shape)`), `actions`, `log_probs`,
        `values`, `advantages`, `returns` (each shape `(T,)`).
        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError("compute_gae() must be called before get()")

        return {
            "observations": np.stack(self.observations).astype(np.float32),
            "actions": np.array(self.actions, dtype=np.int64),
            "log_probs": np.array(self.log_probs, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
            "advantages": self.advantages,
            "returns": self.returns,
        }

    def __len__(self) -> int:
        return len(self.observations)
