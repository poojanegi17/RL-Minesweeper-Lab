"""Core Minesweeper game engine.

This module implements the rules of Minesweeper in isolation: board setup,
mine placement, revealing cells (with cascade reveal of zero-count regions),
and win/loss detection. It has no awareness of reinforcement learning.

Board state uses:
    -1   -> hidden cell
    0-8  -> revealed cell, showing its count of adjacent mines
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple


class Minesweeper:
    """A minimal Minesweeper game engine with a configurable board and mine count."""

    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        num_mines: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        """Create a new game and place mines.

        Args:
            rows: Number of board rows.
            cols: Number of board columns.
            num_mines: Number of mines to place (must be less than rows * cols).
            seed: Optional seed for reproducible mine placement.
        """
        if num_mines >= rows * cols:
            raise ValueError("num_mines must be smaller than the total number of cells")

        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self._rng = random.Random(seed)

        self.mines: List[List[bool]] = []
        self.adjacent_counts: List[List[int]] = []
        self.revealed: List[List[bool]] = []
        self.board: List[List[int]] = []
        self.game_over: bool = False
        self.won: bool = False

        self.reset()

    def reset(self) -> List[List[int]]:
        """Start a new game: re-place mines and clear all revealed cells."""
        self.mines = [[False] * self.cols for _ in range(self.rows)]
        self.revealed = [[False] * self.cols for _ in range(self.rows)]
        self.game_over = False
        self.won = False

        self._place_mines()
        self._compute_adjacent_counts()
        self._update_board()
        return self.board

    def reveal(self, row: int, col: int) -> List[List[int]]:
        """Reveal a cell, cascading through any adjacent zero-count cells.

        No-op if the game has already ended or the cell is already revealed.
        """
        if self.game_over or self.revealed[row][col]:
            return self.board

        if self.mines[row][col]:
            self.revealed[row][col] = True
            self.game_over = True
            self.won = False
            self._update_board()
            return self.board

        self._flood_reveal(row, col)
        self._update_board()

        if self._check_win():
            self.game_over = True
            self.won = True

        return self.board

    def is_game_over(self) -> bool:
        """Return True if the game has ended (win or mine hit)."""
        return self.game_over

    def is_won(self) -> bool:
        """Return True if the game ended in a win."""
        return self.won

    def get_state(self) -> List[List[int]]:
        """Return the current board state (-1 for hidden, 0-8 for revealed)."""
        return self.board

    def _place_mines(self) -> None:
        cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        mine_cells = self._rng.sample(cells, self.num_mines)
        for r, c in mine_cells:
            self.mines[r][c] = True

    def _compute_adjacent_counts(self) -> None:
        self.adjacent_counts = [[0] * self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                if self.mines[r][c]:
                    continue
                self.adjacent_counts[r][c] = sum(
                    1 for nr, nc in self._neighbors(r, c) if self.mines[nr][nc]
                )

    def _neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        result = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    result.append((nr, nc))
        return result

    def _flood_reveal(self, row: int, col: int) -> None:
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if self.revealed[r][c]:
                continue
            self.revealed[r][c] = True
            if self.adjacent_counts[r][c] == 0:
                for nr, nc in self._neighbors(r, c):
                    if not self.revealed[nr][nc] and not self.mines[nr][nc]:
                        stack.append((nr, nc))

    def _update_board(self) -> None:
        self.board = [
            [
                self.adjacent_counts[r][c] if self.revealed[r][c] else -1
                for c in range(self.cols)
            ]
            for r in range(self.rows)
        ]

    def _check_win(self) -> bool:
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.mines[r][c] and not self.revealed[r][c]:
                    return False
        return True
