"""Re-score every (agent, level, density) board-config cell at a higher episode count.

The board/density table's cells were each written by `evaluate_board_config.py`
at its old 200-episode default. At the win rates involved that is 1-3 wins for
several cells and ~9 for others, which is far too coarse to compare densities
against each other -- the same problem the ablation summaries had (see
`reevaluate_checkpoints.py`). This driver re-runs every cell at a single higher
episode count and writes the results into both the full local tree and the
deployment-safe one the backend actually serves.

Q-Learning is only evaluated where the project evaluates it at all -- Beginner.
Its table is keyed by exact board pattern, so a 9x9 or 16x16 state space would
leave nearly every evaluation state unseen; running it there would be a slower
way to reproduce the Random baseline, not a new result (see the README).

Run with:
    python -m evaluation.rescore_board_configs --episodes 2000
    python -m evaluation.rescore_board_configs --episodes 2000 --level beginner
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

from board_configs import BOARD_LEVELS, DENSITY_ORDER, LEVEL_ORDER
from evaluation.evaluate_board_config import Q_TRAIN_EPISODES, SEED, run_evaluation

# Agents evaluated at every level. Q-Learning is handled separately below.
PORTABLE_AGENTS = ["random", "csp", "dqn", "ppo"]
Q_LEARNING_LEVELS = ["beginner"]


def cells(levels: List[str], densities: List[str]) -> List[tuple]:
    """Every (agent, level, density) triple to score, in a stable order."""
    out = []
    for level in levels:
        for density in densities:
            if density not in BOARD_LEVELS[level].densities:
                continue
            for agent in PORTABLE_AGENTS:
                out.append((agent, level, density))
            if level in Q_LEARNING_LEVELS:
                out.append(("q_learning", level, density))
    return out


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the sweep."""
    parser = argparse.ArgumentParser(description="Re-score every board-config cell at a higher episode count.")
    parser.add_argument("--episodes", type=int, default=2000, help="Greedy evaluation episodes per cell.")
    parser.add_argument(
        "--q-train-episodes", type=int, default=Q_TRAIN_EPISODES, help="Training budget for Q-Learning cells."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--level", action="append", default=None, help="Restrict to this level; repeatable. Defaults to all."
    )
    parser.add_argument(
        "--density", action="append", default=None, help="Restrict to this density; repeatable. Defaults to all."
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Where to look up per-level DQN/PPO checkpoints."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        action="append",
        default=None,
        help="Base dir to write {level}/{density}/{agent}_board_result.json into; repeatable. "
        "Defaults to both results_public/levels and results/levels.",
    )
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Score every selected cell and write one result file per output tree."""
    args = parse_args()
    if args.torch_threads is not None:
        import torch

        torch.set_num_threads(args.torch_threads)

    levels = args.level or LEVEL_ORDER
    densities = args.density or DENSITY_ORDER
    output_dirs = [Path(d) for d in (args.output_dir or ["results_public/levels", "results/levels"])]
    todo = cells(levels, densities)

    print(f"Re-scoring {len(todo)} cells at {args.episodes} episodes (seed {args.seed})\n")
    start = time.time()
    for index, (agent, level, density) in enumerate(todo, start=1):
        cell_start = time.time()
        result = run_evaluation(
            agent,
            level,
            density,
            eval_episodes=args.episodes,
            seed=args.seed,
            results_dir=Path(args.results_dir),
            q_train_episodes=args.q_train_episodes,
        )
        for base in output_dirs:
            target = base / level / density
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{agent}_board_result.json").write_text(json.dumps(result, indent=2))

        ci = result["win_rate_ci95"]
        print(
            f"  [{index:>2}/{len(todo)}] {result['agent']:<11} {level:<13} {density:<9} "
            f"{result['win_rate'] * 100:5.2f}%  CI[{ci[0]:.2f}, {ci[1]:.2f}]  "
            f"len={result['avg_episode_length']:5.2f}  ({time.time() - cell_start:.1f}s)"
        )

    print(f"\nDone in {time.time() - start:.1f}s -> {', '.join(str(d) for d in output_dirs)}")


if __name__ == "__main__":
    main()
