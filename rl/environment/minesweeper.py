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

from environment.solvability import is_solvable_without_guessing

# `first_click_safe` policies:
#   "none" -- mines are placed at reset(), before any click, so the opening
#             move can hit a mine. This is the historical behaviour and stays
#             the default: every benchmark this project has published was
#             measured under it, and the RNG draw order is unchanged, so
#             existing seeds reproduce exactly.
#   "cell" -- mines are placed on the first reveal, excluding the clicked
#             cell. The opening move is always survivable.
#   "area" -- as "cell", but the whole 3x3 block around the click is kept
#             mine-free, which forces the opening move to cascade (a cell with
#             no adjacent mines flood-reveals) and hands the player a region of
#             real information instead of a bare number. This is what most
#             desktop Minesweeper implementations do.
FIRST_CLICK_POLICIES = ("none", "cell", "area")


class BoardGenerationError(RuntimeError):
    """Raised when no board satisfying the requested guarantees could be found."""


class Minesweeper:
    """A minimal Minesweeper game engine with a configurable board and mine count."""

    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        num_mines: int = 5,
        seed: Optional[int] = None,
        first_click_safe: str = "none",
        guarantee_solvable: bool = False,
        max_generation_attempts: int = 1000,
    ) -> None:
        """Create a new game and place mines.

        Args:
            rows: Number of board rows.
            cols: Number of board columns.
            num_mines: Number of mines to place (must be less than rows * cols).
            seed: Optional seed for reproducible mine placement.
            first_click_safe: One of `FIRST_CLICK_POLICIES`; see that constant
                for what each policy guarantees. Anything other than "none"
                defers mine placement to the first `reveal` call, so `mines`
                and `adjacent_counts` are all-empty until then.
            guarantee_solvable: If True, resample the mine layout until the
                board can be cleared from the opening click by logical
                deduction alone (see `environment.solvability`), producing
                "no-guess" boards. Requires a `first_click_safe` policy other
                than "none", since solvability is only defined relative to an
                opening move.
            max_generation_attempts: How many layouts to try before giving up
                and raising `BoardGenerationError`. Only used when
                `guarantee_solvable` is set.

        Raises:
            ValueError: If `num_mines` doesn't fit, or the options conflict.
        """
        if num_mines >= rows * cols:
            raise ValueError("num_mines must be smaller than the total number of cells")
        if first_click_safe not in FIRST_CLICK_POLICIES:
            raise ValueError(
                f"Unknown first_click_safe {first_click_safe!r}; choose from {FIRST_CLICK_POLICIES}"
            )
        if guarantee_solvable and first_click_safe == "none":
            raise ValueError(
                "guarantee_solvable requires a first_click_safe policy other than 'none': "
                "whether a board needs a guess depends on which cell is opened first, and a "
                "solvable board cannot have a mine under the opening click"
            )

        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.first_click_safe = first_click_safe
        self.guarantee_solvable = guarantee_solvable
        self.max_generation_attempts = max_generation_attempts
        self._rng = random.Random(seed)

        self.mines: List[List[bool]] = []
        self.adjacent_counts: List[List[int]] = []
        self.revealed: List[List[bool]] = []
        self.board: List[List[int]] = []
        self.game_over: bool = False
        self.won: bool = False
        # Layouts tried while generating the current board. Always 1 unless
        # `guarantee_solvable` had to reject candidates; surfaced through the
        # env's `info` dict so the cost of no-guess generation is measurable.
        self.generation_attempts: int = 0

        self.reset()

    def reset(self) -> List[List[int]]:
        """Start a new game: re-place mines and clear all revealed cells.

        Under a deferred `first_click_safe` policy the mines are *not* placed
        here -- that happens on the first `reveal`, once the opening cell is
        known -- so `mines` and `adjacent_counts` stay all-zero until then.
        """
        self.mines = [[False] * self.cols for _ in range(self.rows)]
        self.adjacent_counts = [[0] * self.cols for _ in range(self.rows)]
        self.revealed = [[False] * self.cols for _ in range(self.rows)]
        self.game_over = False
        self.won = False
        self.generation_attempts = 0

        if self.first_click_safe == "none":
            self._place_mines()
            self._compute_adjacent_counts()
            self.generation_attempts = 1

        self._update_board()
        return self.board

    def reveal(self, row: int, col: int) -> List[List[int]]:
        """Reveal a cell, cascading through any adjacent zero-count cells.

        No-op if the game has already ended or the cell is already revealed.
        Under a deferred `first_click_safe` policy, the very first call is what
        triggers mine placement, so `(row, col)` is guaranteed to be safe.
        """
        if self.game_over or self.revealed[row][col]:
            return self.board

        if self.generation_attempts == 0:
            self._place_mines_around(row, col)

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

    def _forbidden_cells(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Cells the first click's safety policy keeps mine-free."""
        if self.first_click_safe == "area":
            return [(row, col)] + self._neighbors(row, col)
        return [(row, col)]

    def _place_mines_around(self, row: int, col: int) -> None:
        """Place mines for a deferred first click, honouring the active guarantees.

        Args:
            row: Row of the opening click.
            col: Column of the opening click.

        Raises:
            ValueError: If the mines can't fit outside the protected area.
            BoardGenerationError: If `guarantee_solvable` is set and no
                no-guess layout was found within `max_generation_attempts`.
        """
        forbidden = set(self._forbidden_cells(row, col))
        candidates = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in forbidden
        ]
        if self.num_mines > len(candidates):
            raise ValueError(
                f"cannot place {self.num_mines} mines on a {self.rows}x{self.cols} board while "
                f"keeping {len(forbidden)} cells around the first click safe "
                f"(first_click_safe={self.first_click_safe!r}); only {len(candidates)} cells are "
                f"available. Lower the mine count or use first_click_safe='cell'"
            )

        attempts = 0
        while attempts < self.max_generation_attempts:
            attempts += 1
            self.mines = [[False] * self.cols for _ in range(self.rows)]
            for r, c in self._rng.sample(candidates, self.num_mines):
                self.mines[r][c] = True
            self._compute_adjacent_counts()

            if not self.guarantee_solvable or is_solvable_without_guessing(
                self.mines, self.adjacent_counts, self.rows, self.cols, (row, col)
            ):
                self.generation_attempts = attempts
                return

        raise BoardGenerationError(
            f"no no-guess {self.rows}x{self.cols} board with {self.num_mines} mines found in "
            f"{self.max_generation_attempts} attempts. Dense boards are often mostly unsolvable "
            f"by deduction alone -- lower the mine count, use first_click_safe='area' to force an "
            f"opening cascade, or raise max_generation_attempts"
        )

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
