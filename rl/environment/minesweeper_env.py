"""Gymnasium-compatible Minesweeper environment."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.minesweeper import Minesweeper
from environment.utils import action_to_coords, board_to_array

REWARD_SAFE_REVEAL = 1.0
REWARD_MINE_HIT = -10.0
REWARD_WIN = 10.0


class MinesweeperEnv(gym.Env):
    """A Gymnasium environment wrapping the Minesweeper game engine.

    Observation:
        A (rows, cols) int8 numpy array. -1 for hidden cells, 0-8 for
        revealed cells showing their adjacent mine count.

    Action:
        A flattened Discrete(rows * cols) index. Action `a` reveals the
        cell at (a // cols, a % cols).

    Reward:
        +1  for revealing a safe cell
        -10 for hitting a mine
        +10 for winning the game
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        num_mines: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines

        self.game = Minesweeper(rows=rows, cols=cols, num_mines=num_mines, seed=seed)

        self.action_space = spaces.Discrete(rows * cols)
        self.observation_space = spaces.Box(low=-1, high=8, shape=(rows, cols), dtype=np.int8)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return the initial observation."""
        super().reset(seed=seed)

        if seed is not None:
            self.game = Minesweeper(rows=self.rows, cols=self.cols, num_mines=self.num_mines, seed=seed)
        else:
            self.game.reset()

        observation = board_to_array(self.game.get_state())
        info: Dict[str, Any] = {"won": False}
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Reveal the cell corresponding to `action` and return the transition."""
        row, col = action_to_coords(action, self.cols)
        already_revealed = self.game.revealed[row][col]

        self.game.reveal(row, col)

        observation = board_to_array(self.game.get_state())
        terminated = self.game.is_game_over()
        truncated = False
        info: Dict[str, Any] = {"won": self.game.is_won()}

        if already_revealed:
            reward = 0.0
        elif terminated:
            reward = REWARD_WIN if self.game.is_won() else REWARD_MINE_HIT
        else:
            reward = REWARD_SAFE_REVEAL

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Print a human-readable view of the current board to stdout."""
        for row in self.game.get_state():
            print(" ".join("." if v == -1 else str(v) for v in row))
        print()
