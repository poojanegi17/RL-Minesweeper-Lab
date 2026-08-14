"""Deciding whether a Minesweeper board can be cleared by deduction alone.

This backs the engine's `guarantee_solvable` board generation: a candidate
mine layout is accepted only if, starting from the first click, a player can
reach a win without ever guessing.

The deduction rules here deliberately mirror `agents/csp_solver.py` exactly --
the single-constraint rule and the subset rule, applied to a fixpoint, with
*no* global "mines remaining" constraint. That choice is what makes the
guarantee meaningful for this project: "solvable" is defined as "solvable by
the CSP baseline", so a no-guess board set is one on which `CSPAgent` scores
100% by construction, and any shortfall by a learned agent is a shortfall in
deduction rather than luck.

The logic is duplicated rather than imported because `agents` imports from
`environment` (`environment.utils`), so importing back the other way would be
a circular dependency and a layering inversion. If the solver's rule set ever
changes, this module has to change with it or the guarantee silently drifts;
`tests/test_solvability.py` pins the two implementations against each other to
catch that.
"""

from __future__ import annotations

from typing import FrozenSet, List, Sequence, Set, Tuple

Cell = Tuple[int, int]
Constraint = Tuple[FrozenSet[Cell], int]


def neighbors(row: int, col: int, rows: int, cols: int) -> List[Cell]:
    """Return the in-bounds 8-connected neighbors of a cell."""
    result = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                result.append((nr, nc))
    return result


def _effective(
    cells: FrozenSet[Cell],
    count: int,
    known_safe: Set[Cell],
    known_mines: Set[Cell],
) -> Tuple[FrozenSet[Cell], int]:
    """Re-filter a constraint against knowledge learned since it was built."""
    newly_found_mines = cells & known_mines
    remaining = cells - known_safe - known_mines
    return remaining, count - len(newly_found_mines)


def deduce_safe_cells(
    board: Sequence[Sequence[int]],
    rows: int,
    cols: int,
) -> Set[Cell]:
    """Return every hidden cell provably safe from the visible board alone.

    Args:
        board: The player-visible board -- -1 for hidden, 0-8 for a revealed
            cell's adjacent-mine count. Mine positions are never consulted.
        rows: Number of board rows.
        cols: Number of board columns.

    Returns:
        The set of hidden cells that logical deduction proves are not mines.
    """
    known_safe: Set[Cell] = set()
    known_mines: Set[Cell] = set()

    constraints: List[Constraint] = []
    for r in range(rows):
        for c in range(cols):
            value = int(board[r][c])
            if value < 0:
                continue
            hidden_neighbors = [
                (nr, nc) for nr, nc in neighbors(r, c, rows, cols) if board[nr][nc] == -1
            ]
            if hidden_neighbors:
                constraints.append((frozenset(hidden_neighbors), value))

    changed = True
    while changed:
        changed = False
        effective = [
            _effective(cells, count, known_safe, known_mines) for cells, count in constraints
        ]
        effective = [(cells, count) for cells, count in effective if cells]

        for cells, count in effective:
            if count <= 0:
                for cell in cells:
                    if cell not in known_safe:
                        known_safe.add(cell)
                        changed = True
            elif count == len(cells):
                for cell in cells:
                    if cell not in known_mines:
                        known_mines.add(cell)
                        changed = True

        for cells_a, count_a in effective:
            for cells_b, count_b in effective:
                if cells_b == cells_a or not cells_a.issubset(cells_b):
                    continue
                diff = cells_b - cells_a
                diff_count = count_b - count_a
                if diff_count == 0:
                    for cell in diff:
                        if cell not in known_safe:
                            known_safe.add(cell)
                            changed = True
                elif diff_count == len(diff):
                    for cell in diff:
                        if cell not in known_mines:
                            known_mines.add(cell)
                            changed = True

    return {cell for cell in known_safe if board[cell[0]][cell[1]] == -1}


def is_solvable_without_guessing(
    mines: Sequence[Sequence[bool]],
    adjacent_counts: Sequence[Sequence[int]],
    rows: int,
    cols: int,
    first_click: Cell,
) -> bool:
    """Return True if this board can be cleared from `first_click` without guessing.

    Simulates a player who only ever reveals cells that deduction has proven
    safe. If that player clears every non-mine cell, the board needs no guess;
    if they run out of provably-safe moves first, it does.

    Args:
        mines: Ground-truth mine layout, indexed `[row][col]`.
        adjacent_counts: Adjacent-mine count for every cell.
        rows: Number of board rows.
        cols: Number of board columns.
        first_click: The cell the player opens first. Must not be a mine.

    Returns:
        True if deduction alone suffices to win from this opening.
    """
    revealed = [[False] * cols for _ in range(rows)]
    safe_cell_total = rows * cols - sum(1 for r in range(rows) for c in range(cols) if mines[r][c])
    revealed_count = 0

    def flood(row: int, col: int) -> None:
        """Reveal a cell, cascading zero-count regions (mirrors the engine)."""
        nonlocal revealed_count
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if revealed[r][c]:
                continue
            revealed[r][c] = True
            revealed_count += 1
            if adjacent_counts[r][c] == 0:
                for nr, nc in neighbors(r, c, rows, cols):
                    if not revealed[nr][nc] and not mines[nr][nc]:
                        stack.append((nr, nc))

    flood(*first_click)

    while revealed_count < safe_cell_total:
        board = [
            [adjacent_counts[r][c] if revealed[r][c] else -1 for c in range(cols)]
            for r in range(rows)
        ]
        progress = False
        for r, c in deduce_safe_cells(board, rows, cols):
            if not revealed[r][c]:
                flood(r, c)
                progress = True
        if not progress:
            return False

    return True
