"""Simulates several agents taking turns on one *physically shared* Minesweeper
board -- not independent episodes matched by seed (that was the previous,
wrong version of this feature).

Deliberately does not use `MinesweeperEnv.step()`: that (and the underlying
`Minesweeper.reveal()`) assumes a single player whose episode ends the moment
they hit a mine. Here, hitting a mine must only eliminate *that* agent -- the
board and the remaining agents' turns continue. So this module drives
`Minesweeper` directly and handles the mine case itself, *before* ever
calling `reveal()`, so the fatal cell is never written into the shared board
state at all (see module docstring in `generate_race.py` for why that also
happens to be exactly the right call for keeping DQN/PPO's observations
in-distribution, not just a security/simplicity choice).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from environment.minesweeper import Minesweeper
from environment.utils import action_to_coords, board_to_array
from evaluation.replay import ReasoningFn

ActionFn = Callable[[Any], int]
# (display name, action_fn, reasoning_fn) -- reasoning_fn is None for agents
# with nothing to explain (Random), same convention as ReplayRecorder.
AgentEntry = Tuple[str, ActionFn, Optional[ReasoningFn]]


def _to_plain_board(board: List[List[int]]) -> List[List[int]]:
    """`game.get_state()` already returns plain nested ints, but this pins
    that down explicitly rather than trusting it stays that way forever."""
    return [[int(v) for v in row] for row in board]


def simulate_shared_race(game: Minesweeper, agents: List[AgentEntry]) -> Dict[str, Any]:
    """Play one shared game to completion, turn by turn, in `agents`' order.

    Each round, every still-alive agent gets exactly one turn (in the order
    given); a mine hit eliminates that agent (removed from all future
    rounds) without revealing the cell. The game ends when the board is
    collectively cleared (`game.is_won()`, true regardless of *whose* turn
    cleared it) or when every agent has been eliminated -- both are real,
    valid outcomes, not errors.

    Returns a dict with `initial_board`, `turns` (one entry per turn taken),
    `won`, `total_turns`, `surviving_agents`, and `eliminated_agents`
    (agent name -> the turn number that eliminated them).
    """
    initial_board = _to_plain_board(game.get_state())
    alive: List[AgentEntry] = list(agents)
    eliminated_at: Dict[str, int] = {}
    turns: List[Dict[str, Any]] = []
    turn_number = 0

    while alive and not game.is_won():
        for entry in list(alive):
            if game.is_won():
                break
            name, action_fn, reasoning_fn = entry
            turn_number += 1

            observation = board_to_array(game.get_state())
            action = action_fn(observation)
            row, col = action_to_coords(action, game.cols)
            reasoning = reasoning_fn(observation, action) if reasoning_fn is not None else None

            is_mine = game.mines[row][col]
            if is_mine:
                eliminated_at[name] = turn_number
                alive.remove(entry)
            else:
                game.reveal(row, col)

            turns.append(
                {
                    "turn": turn_number,
                    "agent": name,
                    "action": {"row": row, "col": col},
                    "board_state": _to_plain_board(game.get_state()),
                    "eliminated": is_mine,
                    "reasoning": reasoning,
                }
            )

    return {
        "initial_board": initial_board,
        "turns": turns,
        "won": game.is_won(),
        "total_turns": turn_number,
        "surviving_agents": [name for name, _, _ in agents if name not in eliminated_at],
        "eliminated_agents": eliminated_at,
    }
