"""Measure how much of the state space tabular Q-Learning actually re-visits.

Why this exists
---------------
Q-Learning's win rate on this project's boards swings from 82% to 0.05% with no
change to the algorithm, only to the board. The explanation is state repetition,
and this script measures it rather than asserting it.

The mechanism worth keeping in mind while reading the output: `QLearningAgent`
keys its table on the exact flattened board, and `_q_values` returns a fresh
all-zero row for a state it has never seen. So on an unvisited board every legal
move ties at 0 and `choose_action`'s greedy branch picks uniformly at random --
an unseen state makes this agent behave exactly like `RandomAgent`. A run's win
rate is therefore bounded by how much of the game it has seen often enough to
have non-zero, meaningfully-ordered values.

Reported per board:
    - distinct states updated, and the share updated exactly once
    - mean updates per state
    - how many states reach 10+ updates ("well-learned"), what share of all
      updates land there, and their median *depth* (revealed-cell count) --
      depth is the interesting one, since it says how far into a game the
      learned region actually extends
    - the all-hidden opening state's visit count, and the spread of its
      Q-values (it is visited every single episode, so it is always learned)

Run with:
    python -m evaluation.analyze_q_learning_coverage
    python -m evaluation.analyze_q_learning_coverage --episodes 5000
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from agents.q_learning_agent import QLearningAgent
from environment.minesweeper_env import MinesweeperEnv
from evaluation.evaluate_board_config import env_version

SEED = 42
WELL_LEARNED_MIN_VISITS = 10

# (label, rows, cols, mines). The 4x4/2 board is not part of the project's
# level catalog -- it's included because it's the control case that shows the
# algorithm works when states repeat.
BOARDS: List[Tuple[str, int, int, int]] = [
    ("4x4 / 2 mines", 4, 4, 2),
    ("5x5 / 3 mines (sparse)", 5, 5, 3),
    ("5x5 / 5 mines (standard)", 5, 5, 5),
    ("5x5 / 8 mines (dense)", 5, 5, 8),
]


def measure(
    rows: int,
    cols: int,
    mines: int,
    episodes: int,
    seed: int = SEED,
    first_click_safe: str = "none",
) -> Dict[str, Any]:
    """Train a table and return its coverage statistics.

    Visit counts are collected by wrapping `update_q_value`, so they count real
    learning updates only -- not the incidental table inserts `_q_values`
    performs when merely *reading* an unseen state (as a bootstrap target, or
    during evaluation), which would otherwise inflate the table size.
    """
    env = MinesweeperEnv(
        rows=rows, cols=cols, num_mines=mines, seed=seed, first_click_safe=first_click_safe
    )
    agent = QLearningAgent(rows=rows, cols=cols, seed=seed)

    counts: collections.Counter = collections.Counter()
    original_update = agent.update_q_value

    def counting_update(state, action, reward, next_state, done):  # type: ignore[no-untyped-def]
        counts[agent.encode_state(state)] += 1
        original_update(state, action, reward, next_state, done)

    agent.update_q_value = counting_update  # type: ignore[method-assign]
    agent.train(env, episodes=episodes)

    total_updates = sum(counts.values())
    updated_once = sum(1 for v in counts.values() if v == 1)
    well_learned = {s: v for s, v in counts.items() if v >= WELL_LEARNED_MIN_VISITS}
    # A state's "depth" is how many of its cells are revealed -- i.e. how far
    # into an episode it occurs.
    depths = [sum(1 for cell in state if cell != -1) for state in well_learned]

    opening = tuple([-1] * (rows * cols))
    opening_q = agent.q_table.get(opening)

    return {
        "rows": rows,
        "cols": cols,
        "mines": mines,
        "safe_cells": rows * cols - mines,
        "episodes": episodes,
        "states_updated": len(counts),
        "total_updates": total_updates,
        "updated_once_pct": round(updated_once / len(counts) * 100, 2),
        "mean_updates_per_state": round(total_updates / len(counts), 3),
        "well_learned_states": len(well_learned),
        "well_learned_update_share_pct": round(sum(well_learned.values()) / total_updates * 100, 2),
        "well_learned_median_depth": statistics.median(depths) if depths else None,
        "opening_visits": counts[opening],
        "opening_q_spread": round(float(opening_q.max() - opening_q.min()), 4) if opening_q is not None else None,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Measure tabular Q-Learning's state-space coverage.")
    parser.add_argument(
        "--episodes", type=int, default=20000, help="Training episodes per board (matches evaluate_agents.py)."
    )
    parser.add_argument(
        "--first-click-safe",
        choices=["none", "cell", "area"],
        default="none",
        help="Board-generation policy. Changes the state distribution the table sees, so coverage "
        "measured under one value is not comparable to another (default: none, matching v1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Defaults to analysis/q_learning_coverage_{env_version}.json.",
    )
    return parser.parse_args()


def main() -> None:
    """Measure every board and write a JSON report."""
    args = parse_args()
    version = env_version(args.first_click_safe, False)
    output = Path(args.output) if args.output else Path("analysis") / f"q_learning_coverage_{version}.json"
    report: Dict[str, Any] = {
        "episodes": args.episodes,
        "seed": SEED,
        "env_version": version,
        "first_click_safe": args.first_click_safe,
        "boards": {},
    }

    print(f"Q-Learning state-space coverage after {args.episodes:,} training episodes (seed {SEED})\n")
    header = ("board", "states", "updates", "%once", "mean", ">=10x", "share", "med depth", "open Q spread")
    print("%-26s %8s %9s %7s %7s %7s %7s %10s %14s" % header)
    for label, rows, cols, mines in BOARDS:
        stats = measure(rows, cols, mines, args.episodes, first_click_safe=args.first_click_safe)
        report["boards"][label] = stats
        print(
            "%-26s %8s %9s %6.1f%% %7.2f %7s %6.1f%% %10s %14s"
            % (
                label,
                f"{stats['states_updated']:,}",
                f"{stats['total_updates']:,}",
                stats["updated_once_pct"],
                stats["mean_updates_per_state"],
                f"{stats['well_learned_states']:,}",
                stats["well_learned_update_share_pct"],
                f"{stats['well_learned_median_depth']} of {stats['safe_cells']}",
                stats["opening_q_spread"],
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {output}")
    print(
        "\nRead 'med depth' as how far into a game the learned region reaches: a depth of 2 on a\n"
        "20-safe-cell board means the table only ever learned the opening couple of moves, and\n"
        "everything after that is an unseen state where greedy selection is uniform random."
    )


if __name__ == "__main__":
    main()
