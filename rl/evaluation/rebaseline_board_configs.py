"""Re-measure every baseline agent across the whole board catalog under one
set of environment options, writing to a version-scoped results tree.

Why this exists: `first_click_safe` and `guarantee_solvable` change the board
*distribution*, so they move every agent's numbers -- including Random and CSP,
which never consult reward and are otherwise invariant to environment changes.
A DQN result measured on the new distribution cannot be read against a CSP
ceiling or a Random floor measured on the old one; the comparison would be
between two different games. So the floor and the ceiling get re-measured
together, in one pass, before any learned agent is scored.

Only agents that need no pre-existing checkpoint are run by default (Random,
CSP, Q-Learning). DQN and PPO have to be *retrained* under the new environment
before they can be re-evaluated -- their committed checkpoints were fitted to
the old board distribution -- so they are opt-in via `--agents` and will fail
loudly if no matching checkpoint exists.

Run with:
    python -m evaluation.rebaseline_board_configs --first-click-safe area
    python -m evaluation.rebaseline_board_configs --first-click-safe area --levels beginner
    python -m evaluation.rebaseline_board_configs --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from board_configs import BOARD_LEVELS, DENSITY_ORDER, LEVEL_ORDER
from environment.minesweeper import BoardGenerationError
from evaluation.evaluate_board_config import (
    EVAL_EPISODES,
    Q_TRAIN_EPISODES,
    SEED,
    env_version,
    run_evaluation,
)

# Agents needing no checkpoint, so a re-baseline can always produce them.
# Q-Learning trains its own table inside `run_evaluation`; Random and CSP need
# no training at all.
BASELINE_AGENTS = ("random", "csp", "q_learning")

# Q-Learning's table is keyed by exact board pattern, so it neither generalizes
# nor fits in memory beyond the smallest boards -- which is why the committed
# results only include it at "beginner". Restricting it here keeps the
# re-baseline matched to what already exists rather than inventing new cells.
Q_LEARNING_LEVELS = ("beginner",)


def cells(levels: List[str], densities: List[str]) -> List[tuple]:
    """Every (level, density) pair to evaluate, in catalog order."""
    return [
        (level, density)
        for level in levels
        for density in densities
        if density in BOARD_LEVELS[level].densities
    ]


def applies(agent: str, level: str) -> bool:
    """Whether `agent` is measured at `level` (see `Q_LEARNING_LEVELS`)."""
    return not (agent == "q_learning" and level not in Q_LEARNING_LEVELS)


def run(
    agents: List[str],
    levels: List[str],
    densities: List[str],
    output_dir: Path,
    *,
    first_click_safe: str,
    guarantee_solvable: bool,
    eval_episodes: int,
    q_train_episodes: int,
    seed: int,
    results_dir: Path,
    checkpoint_experiment: Optional[str],
    dry_run: bool,
) -> int:
    """Evaluate every applicable (agent, level, density) and write the results.

    Returns:
        A process exit code: 0 if everything requested succeeded, 1 otherwise.
    """
    version = env_version(first_click_safe, guarantee_solvable)
    planned = [
        (agent, level, density)
        for level, density in cells(levels, densities)
        for agent in agents
        if applies(agent, level)
    ]

    print(f"env_version={version}  first_click_safe={first_click_safe!r}  guarantee_solvable={guarantee_solvable}")
    print(f"{len(planned)} evaluations -> {output_dir}\n")
    if dry_run:
        for agent, level, density in planned:
            print(f"  would evaluate {agent:10} @ {level}/{density}")
        return 0

    failures: List[str] = []
    for index, (agent, level, density) in enumerate(planned, start=1):
        label = f"{agent} @ {level}/{density}"
        started = time.time()
        try:
            result = run_evaluation(
                agent,
                level,
                density,
                eval_episodes=eval_episodes,
                seed=seed,
                results_dir=results_dir,
                q_train_episodes=q_train_episodes,
                first_click_safe=first_click_safe,
                guarantee_solvable=guarantee_solvable,
                checkpoint_experiment=checkpoint_experiment,
            )
        except BoardGenerationError as exc:
            # Expected at high densities under --guarantee-solvable: most dense
            # layouts genuinely need a guess, so no amount of resampling finds
            # one. Recorded and skipped rather than aborting the whole sweep.
            failures.append(f"{label}: board generation failed ({exc})")
            print(f"[{index}/{len(planned)}] {label:34} SKIPPED -- no solvable board")
            continue
        except (FileNotFoundError, KeyError) as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"[{index}/{len(planned)}] {label:34} FAILED -- {exc}")
            continue

        destination = output_dir / level / density
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"{agent}_board_result.json"
        path.write_text(json.dumps(result, indent=2))

        print(
            f"[{index}/{len(planned)}] {label:34} win_rate={result['win_rate'] * 100:5.2f}%  "
            f"len={result['avg_episode_length']:5.2f}  ({time.time() - started:.0f}s)"
        )

    print()
    if failures:
        print(f"{len(failures)} of {len(planned)} evaluations did not produce a result:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"All {len(planned)} evaluations written under {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-measure baseline agents across the board catalog under one environment version."
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=list(BASELINE_AGENTS),
        choices=["random", "csp", "q_learning", "dqn", "ppo"],
        help="Agents to evaluate. DQN/PPO need a checkpoint trained under the same environment "
        "options; their committed checkpoints were fitted to the old board distribution.",
    )
    parser.add_argument("--levels", nargs="+", default=list(LEVEL_ORDER), choices=list(LEVEL_ORDER))
    parser.add_argument("--densities", nargs="+", default=list(DENSITY_ORDER), choices=list(DENSITY_ORDER))
    parser.add_argument(
        "--first-click-safe",
        choices=["none", "cell", "area"],
        default="area",
        help="Defaults to 'area' -- the recommended benchmark distribution, which keeps forced "
        "guesses in the game so CSP stays a meaningful ceiling.",
    )
    parser.add_argument(
        "--guarantee-solvable",
        action="store_true",
        help="No-guess boards. Intended for training curricula; CSP scores 100%% by construction, "
        "so this is not a benchmark setting.",
    )
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--q-train-episodes", type=int, default=Q_TRAIN_EPISODES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--results-dir", type=str, default="results", help="Where to look up checkpoints from.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Defaults to results_public/{env_version}/levels, keeping new results in a tree the "
        "backend does not read until pointed at it.",
    )
    parser.add_argument(
        "--checkpoint-experiment",
        type=str,
        default=None,
        help="Directory under --results-dir holding the DQN/PPO checkpoint to score, overriding "
        "automatic resolution. Applies to every selected cell, so pair it with a single --agents "
        "and --levels selection. Without it, beginner-level DQN/PPO resolve to this project's "
        "original v1 checkpoint, which would report stale transfer as a v2 result.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List the planned evaluations and exit.")
    parser.add_argument("--torch-threads", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    version = env_version(args.first_click_safe, args.guarantee_solvable)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("results_public") / version / "levels"
    )

    raise SystemExit(
        run(
            args.agents,
            args.levels,
            args.densities,
            output_dir,
            first_click_safe=args.first_click_safe,
            guarantee_solvable=args.guarantee_solvable,
            eval_episodes=args.eval_episodes,
            q_train_episodes=args.q_train_episodes,
            seed=args.seed,
            results_dir=Path(args.results_dir),
            checkpoint_experiment=args.checkpoint_experiment,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
