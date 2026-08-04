"""A simple random-action baseline agent for the Minesweeper environment."""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

import numpy as np


class RandomAgent:
    """An agent that selects a random hidden cell on every step."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, observation: np.ndarray) -> int:
        """Pick a random action among cells that are still hidden (-1)."""
        hidden_indices = np.flatnonzero(observation.flatten() == -1)
        return int(self._rng.choice(hidden_indices.tolist()))

    def play_episode(self, env, render: bool = False) -> Dict[str, Any]:
        """Play one full episode using random valid actions until termination.

        Returns a summary dict with total reward, step count, and whether
        the episode ended in a win.
        """
        observation, info = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = self.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if render:
                env.render()

        return {
            "total_reward": total_reward,
            "steps": steps,
            "won": info.get("won", False),
        }
