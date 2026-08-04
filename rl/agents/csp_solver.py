"""Constraint Satisfaction Problem (CSP) based Minesweeper solver.

Each revealed numbered cell defines a constraint: "exactly N of these hidden
neighbor cells are mines." The solver applies two logical deduction rules to
a fixpoint:

    1. Single-constraint rule: if a constraint's mine count is 0, every cell
       in it is safe; if the count equals the number of cells, every cell is
       a mine.
    2. Subset rule: if constraint A's cells are a subset of constraint B's
       cells, the leftover cells (B - A) must contain exactly
       `B.count - A.count` mines, which is often enough to resolve them too.

When no further deduction is possible, the solver falls back to picking the
hidden cell with the lowest estimated mine probability, or a random hidden
cell if no information is available at all (e.g. the first move).
"""

from __future__ import annotations

import random
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from environment.utils import coords_to_action

Cell = Tuple[int, int]
Constraint = Tuple[FrozenSet[Cell], int]


class CSPAgent:
    """A logical-deduction baseline agent for Minesweeper."""

    def __init__(
        self,
        rows: int,
        cols: int,
        num_mines: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Create a solver for a board of the given size.

        Args:
            rows: Number of board rows.
            cols: Number of board columns.
            num_mines: Total mines on the board, if known. Used only to
                estimate probabilities for cells with no direct constraint.
            seed: Optional seed for reproducible tie-breaking/fallback choices.
        """
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self._rng = random.Random(seed)

        self.known_safe: Set[Cell] = set()
        self.known_mines: Set[Cell] = set()
        self.constraints: List[Constraint] = []

    def reset(self) -> None:
        """Clear all deduced knowledge. Call at the start of each new episode."""
        self.known_safe.clear()
        self.known_mines.clear()
        self.constraints = []

    def choose_action(self, state: Sequence[Sequence[int]]) -> int:
        """Pick the next cell to reveal, as a flattened action index."""
        self.update_constraints(state)

        safe_moves = self.find_safe_moves(state)
        if safe_moves:
            row, col = safe_moves[self._rng.randrange(len(safe_moves))]
            return coords_to_action(row, col, self.cols)

        probabilities = self.calculate_probabilities(state)
        if probabilities:
            best_prob = min(probabilities.values())
            best_cells = [cell for cell, prob in probabilities.items() if prob == best_prob]
            row, col = best_cells[self._rng.randrange(len(best_cells))]
            return coords_to_action(row, col, self.cols)

        hidden = [
            (r, c) for r in range(self.rows) for c in range(self.cols) if state[r][c] == -1
        ]
        row, col = self._rng.choice(hidden)
        return coords_to_action(row, col, self.cols)

    def update_constraints(self, state: Sequence[Sequence[int]]) -> None:
        """Rebuild constraints from the board and deduce until no new info is found."""
        self.constraints = []
        for r in range(self.rows):
            for c in range(self.cols):
                value = int(state[r][c])
                if value < 0:
                    continue

                hidden_neighbors = [
                    (nr, nc) for nr, nc in self._neighbors(r, c) if state[nr][nc] == -1
                ]
                unresolved = [
                    cell
                    for cell in hidden_neighbors
                    if cell not in self.known_safe and cell not in self.known_mines
                ]
                known_mine_neighbors = sum(1 for cell in hidden_neighbors if cell in self.known_mines)
                remaining_count = value - known_mine_neighbors

                if unresolved:
                    self.constraints.append((frozenset(unresolved), remaining_count))

        self._deduce()

    def find_safe_moves(self, state: Sequence[Sequence[int]]) -> List[Cell]:
        """Return known-safe cells that are still hidden on the given board."""
        return [cell for cell in self.known_safe if state[cell[0]][cell[1]] == -1]

    def find_mines(self) -> Set[Cell]:
        """Return the set of cells deduced to be mines."""
        return set(self.known_mines)

    def calculate_probabilities(self, state: Sequence[Sequence[int]]) -> Dict[Cell, float]:
        """Estimate mine probability for every unresolved hidden cell.

        Cells touched by one or more constraints get the average of each
        constraint's naive probability (count / size). Cells with no
        constraint at all fall back to a global estimate based on
        `num_mines`, or 0.5 if the total mine count is unknown.
        """
        hidden_cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if state[r][c] == -1 and (r, c) not in self.known_safe and (r, c) not in self.known_mines
        ]
        if not hidden_cells:
            return {}

        per_cell_estimates: Dict[Cell, List[float]] = {}
        for cells, count in self.constraints:
            remaining_cells, effective_count = self._effective(cells, count)
            if not remaining_cells:
                continue
            probability = max(0.0, min(1.0, effective_count / len(remaining_cells)))
            for cell in remaining_cells:
                per_cell_estimates.setdefault(cell, []).append(probability)

        if self.num_mines is not None:
            remaining_mines = max(0, self.num_mines - len(self.known_mines))
            fallback_probability = remaining_mines / len(hidden_cells)
        else:
            fallback_probability = 0.5

        return {
            cell: (sum(per_cell_estimates[cell]) / len(per_cell_estimates[cell]))
            if cell in per_cell_estimates
            else fallback_probability
            for cell in hidden_cells
        }

    def _neighbors(self, row: int, col: int) -> List[Cell]:
        result = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    result.append((nr, nc))
        return result

    def _effective(self, cells: FrozenSet[Cell], count: int) -> Tuple[FrozenSet[Cell], int]:
        """Re-filter a constraint's cells against knowledge learned since it was built."""
        newly_found_mines = cells & self.known_mines
        remaining = cells - self.known_safe - self.known_mines
        return remaining, count - len(newly_found_mines)

    def _deduce(self) -> None:
        """Apply the single-constraint and subset rules until reaching a fixpoint."""
        changed = True
        while changed:
            changed = False
            effective = [self._effective(cells, count) for cells, count in self.constraints]
            effective = [(cells, count) for cells, count in effective if cells]

            for cells, count in effective:
                if count <= 0:
                    for cell in cells:
                        if cell not in self.known_safe:
                            self.known_safe.add(cell)
                            changed = True
                elif count == len(cells):
                    for cell in cells:
                        if cell not in self.known_mines:
                            self.known_mines.add(cell)
                            changed = True

            for cells_a, count_a in effective:
                for cells_b, count_b in effective:
                    if cells_b == cells_a or not cells_a.issubset(cells_b):
                        continue
                    diff = cells_b - cells_a
                    diff_count = count_b - count_a
                    if diff_count == 0:
                        for cell in diff:
                            if cell not in self.known_safe:
                                self.known_safe.add(cell)
                                changed = True
                    elif diff_count == len(diff):
                        for cell in diff:
                            if cell not in self.known_mines:
                                self.known_mines.add(cell)
                                changed = True
