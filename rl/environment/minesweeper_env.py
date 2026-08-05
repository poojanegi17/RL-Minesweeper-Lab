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

# "shaped" reward_mode constants. Every prior DQN/PPO benchmark in this
# project was measured under the constants above, so those stay the
# untouched default; "shaped" is an opt-in alternative, not a replacement.
#
# Reasoning (see also the README's PPO section, which motivated this):
#   - SAFE_REVEAL stays +1, same as default -- shaping adds to the existing
#     signal rather than replacing its baseline case.
#   - CASCADE_BONUS_PER_CELL rewards *additional* cells revealed by a single
#     action's flood-fill (see `Minesweeper._flood_reveal`). A cascade only
#     happens when the clicked cell's adjacent mine count is 0, which is
#     public information derivable from the resulting board -- never from
#     `self.game.mines` -- so this remains a legal, mine-location-blind
#     signal. It's denser and more informative than a flat +1 per reveal
#     regardless of how much of the board that reveal actually resolved,
#     directly targeting the "sparse rewards / poor credit assignment"
#     bottleneck identified after PPO's initial 0% benchmark.
#   - MINE_HIT and WIN are scaled up (in magnitude) relative to
#     SAFE_REVEAL, sharpening the gap between the terminal outcomes that
#     actually decide the game and the per-step reveal reward that
#     otherwise dominates by sheer frequency early in training.
SHAPED_REWARD_SAFE_REVEAL = 1.0
SHAPED_REWARD_CASCADE_BONUS_PER_CELL = 0.2
SHAPED_REWARD_MINE_HIT = -15.0
SHAPED_REWARD_WIN = 20.0

REWARD_MODES = ("default", "shaped")


class MinesweeperEnv(gym.Env):
    """A Gymnasium environment wrapping the Minesweeper game engine.

    Observation:
        A (rows, cols) int8 numpy array. -1 for hidden cells, 0-8 for
        revealed cells showing their adjacent mine count.

    Action:
        A flattened Discrete(rows * cols) index. Action `a` reveals the
        cell at (a // cols, a % cols).

    Reward (reward_mode="default", the default):
        +1  for revealing a safe cell
        -10 for hitting a mine
        +10 for winning the game

    Reward (reward_mode="shaped"):
        +1 (+ 0.2 per additional cell auto-revealed by the same action's
            cascade) for revealing a safe cell
        -15 for hitting a mine
        +20 for winning the game
        See the `SHAPED_REWARD_*` constants above for the reasoning. Only
        ever derived from the resulting board and win/loss outcome -- never
        from hidden mine positions -- so it doesn't relax the "agent only
        sees what the environment returns" fairness rule any agent in this
        project follows.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        num_mines: int = 5,
        seed: Optional[int] = None,
        reward_mode: str = "default",
    ) -> None:
        super().__init__()
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"Unknown reward_mode {reward_mode!r}; choose from {REWARD_MODES}")

        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.reward_mode = reward_mode

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
        revealed_before = sum(sum(r) for r in self.game.revealed)

        self.game.reveal(row, col)

        observation = board_to_array(self.game.get_state())
        terminated = self.game.is_game_over()
        truncated = False
        info: Dict[str, Any] = {"won": self.game.is_won()}

        shaped = self.reward_mode == "shaped"
        if already_revealed:
            reward = 0.0
        elif terminated:
            if self.game.is_won():
                reward = SHAPED_REWARD_WIN if shaped else REWARD_WIN
            else:
                reward = SHAPED_REWARD_MINE_HIT if shaped else REWARD_MINE_HIT
        elif shaped:
            revealed_after = sum(sum(r) for r in self.game.revealed)
            cascade_extra_cells = max(revealed_after - revealed_before - 1, 0)
            reward = SHAPED_REWARD_SAFE_REVEAL + cascade_extra_cells * SHAPED_REWARD_CASCADE_BONUS_PER_CELL
        else:
            reward = REWARD_SAFE_REVEAL

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Print a human-readable view of the current board to stdout."""
        for row in self.game.get_state():
            print(" ".join("." if v == -1 else str(v) for v in row))
        print()
