"""Tabular Q-Learning agent for Minesweeper.

The visible board (-1 for hidden, 0-8 for revealed) is used directly as the
state: each distinct board pattern is a Q-table key. Because the environment
re-randomizes mine placement on every `reset()`, a given state is reachable
under many different underlying mine layouts, so the learned Q(s, a) is the
expected return for taking action `a` in board-pattern `s`, averaged over the
environment's mine-placement distribution -- an ordinary stochastic MDP, just
with a very large state space. This agent only ever reads the observation
returned by the environment; it never inspects `env.game.mines`.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

State = Tuple[int, ...]


class QLearningAgent:
    """A tabular Q-learning agent that treats each board pattern as a state."""

    def __init__(
        self,
        rows: int,
        cols: int,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        """Create a Q-learning agent for a board of the given size.

        Args:
            rows: Number of board rows.
            cols: Number of board columns.
            alpha: Learning rate.
            gamma: Discount factor for future rewards.
            epsilon: Initial exploration probability.
            epsilon_min: Floor that epsilon decays towards.
            epsilon_decay: Multiplicative decay applied to epsilon after each
                training episode.
            seed: Optional seed for reproducible action selection.
        """
        self.rows = rows
        self.cols = cols
        self.n_actions = rows * cols

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self._rng = random.Random(seed)
        self.q_table: Dict[State, np.ndarray] = {}

    def encode_state(self, observation: Sequence[Sequence[int]]) -> State:
        """Convert a board observation into a hashable Q-table key.

        The board is flattened row-major into a plain tuple of ints, which is
        both hashable (usable as a dict key) and independent of whether the
        caller passed a numpy array or nested lists.
        """
        return tuple(int(v) for v in np.asarray(observation).flatten())

    def choose_action(self, observation: Sequence[Sequence[int]], explore: bool = True) -> int:
        """Pick an action for `observation` using epsilon-greedy selection.

        Only hidden cells are considered, since revealing an already-revealed
        cell is always a wasted move (reward 0, no state change). When
        `explore` is False the agent acts greedily with respect to the
        current Q-values, which is what evaluation should use.
        """
        state = self.encode_state(observation)
        hidden_actions = self._hidden_actions(state)
        if not hidden_actions:
            return self._rng.randrange(self.n_actions)

        if explore and self._rng.random() < self.epsilon:
            return self._rng.choice(hidden_actions)

        q_values = self._q_values(state)
        best_q = max(q_values[a] for a in hidden_actions)
        best_actions = [a for a in hidden_actions if q_values[a] == best_q]
        return self._rng.choice(best_actions)

    def select_action(self, observation: Sequence[Sequence[int]]) -> int:
        """Greedy action selection with no exploration, for use in evaluation."""
        return self.choose_action(observation, explore=False)

    def update_q_value(
        self,
        state: Sequence[Sequence[int]],
        action: int,
        reward: float,
        next_state: Sequence[Sequence[int]],
        done: bool,
    ) -> None:
        """Apply one Q-learning update:

            Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))

        The next-state max is taken only over cells still hidden in `next_state`
        (mirroring `choose_action`'s masking), and dropped entirely when `done`
        is True since there is no future return to bootstrap from.
        """
        state_key = self.encode_state(state)
        q_values = self._q_values(state_key)

        if done:
            target = reward
        else:
            next_key = self.encode_state(next_state)
            next_hidden = self._hidden_actions(next_key)
            if next_hidden:
                next_q = self._q_values(next_key)
                best_next = max(next_q[a] for a in next_hidden)
            else:
                best_next = 0.0
            target = reward + self.gamma * best_next

        q_values[action] += self.alpha * (target - q_values[action])

    def decay_epsilon(self) -> None:
        """Decay epsilon towards `epsilon_min` after a training episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_value(self, observation: Sequence[Sequence[int]], action: int) -> float:
        """Return the current Q-value for `(observation, action)`."""
        return float(self._q_values(self.encode_state(observation))[action])

    def train(self, env: Any, episodes: int) -> List[Dict[str, Any]]:
        """Run `episodes` training episodes, updating Q-values after every step.

        Returns a per-episode history of total reward, step count, win flag,
        and the epsilon value used during that episode -- useful for plotting
        or printing a training curve.
        """
        history: List[Dict[str, Any]] = []

        for _ in range(episodes):
            observation, info = env.reset()
            terminated = truncated = False
            total_reward = 0.0
            steps = 0

            while not (terminated or truncated):
                action = self.choose_action(observation, explore=True)
                next_observation, reward, terminated, truncated, info = env.step(action)
                self.update_q_value(observation, action, reward, next_observation, terminated or truncated)
                observation = next_observation
                total_reward += reward
                steps += 1

            history.append(
                {
                    "total_reward": total_reward,
                    "steps": steps,
                    "won": info.get("won", False),
                    "epsilon": self.epsilon,
                }
            )
            self.decay_epsilon()

        return history

    def _hidden_actions(self, state: State) -> List[int]:
        return [i for i, v in enumerate(state) if v == -1]

    def _q_values(self, state: State) -> np.ndarray:
        q_values = self.q_table.get(state)
        if q_values is None:
            q_values = np.zeros(self.n_actions, dtype=np.float64)
            self.q_table[state] = q_values
        return q_values
