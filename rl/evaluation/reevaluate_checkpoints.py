"""Re-evaluate every saved DQN/PPO checkpoint at a higher episode count.

Why this script exists
----------------------
Every experiment summary in this project reports `win_rate` from a 200-episode
greedy evaluation (`EVAL_EPISODES` in `dqn_experiment.py`/`ppo_experiment.py`).
On this benchmark a good agent wins 1-3% of games, so 200 episodes means the
reported number rests on *2 to 7 wins*. At that sample size the 95% confidence
interval spans a factor of ~5, and every pairwise difference between the
project's ablation variants was statistically indistinguishable from zero
(pairwise Fisher exact p between 0.18 and 0.75). Three of the five DQN
Beginner conclusions changed once the same checkpoints were re-scored at
2,000 episodes.

Two further biases this script exists to correct:

- **Deployment protocol was confounded with the training variable.** Some
  experiments evaluated `final_model.pt` and others `best_model.pt`, so
  "variant X beat variant Y" mixed a training change together with a
  *which-weights-to-deploy* change. This script always scores **both** files
  for every run, so the two effects can be read separately.
- **Every model is scored on the same boards.** The eval seed is fixed for a
  whole report and a fresh env is constructed per model, so all runs face an
  identical episode sequence. It defaults to `EVAL_SEED`, which is deliberately
  *not* the seed the experiments themselves used -- a held-out board set. Pass
  `--eval-seed 42` to score on the same boards as the published summaries
  instead, which makes these numbers comparable to those as well as to each
  other. The original per-experiment evaluations reused the training env
  whose RNG had already advanced, making the board set an unrecorded
  function of training length.

Output is a single JSON report; it never modifies an existing summary file.

Run with:
    python -m evaluation.reevaluate_checkpoints --episodes 2000
    python -m evaluation.reevaluate_checkpoints --episodes 2000 --only exp_
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from environment.minesweeper_env import MinesweeperEnv

# Held identical across every model scored in a single report, so any
# difference between two rows is a difference between the models and not
# between the boards they happened to face.
# The historical default for this report. `--eval-seed 42` instead matches
# the seed every experiment summary is scored on, making a re-scored number
# directly comparable to the run's own published figure rather than merely
# comparable to the other rows here.
EVAL_SEED = 4242

# Board fallback for the earliest Beginner summaries, which predate
# `rows`/`cols`/`mines` being written into the summary JSON at all. Matches
# the same 5x5/5-mine benchmark constant `backend/app/config.py` falls back to.
DEFAULT_BOARD = (5, 5, 5)

DQN_CHECKPOINTS = {"final": "final_model.pt", "best": "best_model.pt"}
PPO_CHECKPOINTS = {"final": "final_policy.pt", "best": "best_policy.pt"}


def wilson_interval(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion, as percentages.

    Preferred over the normal approximation here because win rates near 1%
    with n in the low thousands put the normal interval's lower bound below
    zero, which is not a meaningful bound for a count.
    """
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # Clamped because at `wins == 0` the two terms cancel only up to floating
    # point, leaving a tiny negative lower bound that would render as "-0.00%".
    # A bound on a count is never outside [0, 100].
    return (max(0.0, (centre - half) * 100), min(100.0, (centre + half) * 100))


def two_proportion_test(wins_a: int, wins_b: int, n: int) -> Tuple[float, float]:
    """Pooled two-proportion z-test; returns `(z, two_sided_p)`.

    Used only for same-`n` comparisons between two rows of one report, which
    is the only comparison this script's output supports.
    """
    if n == 0:
        return (0.0, 1.0)
    pooled = (wins_a + wins_b) / (2 * n)
    se = math.sqrt(pooled * (1 - pooled) * 2 / n)
    if se == 0:
        return (0.0, 1.0)
    z = (wins_b - wins_a) / n / se
    return (z, math.erfc(abs(z) / math.sqrt(2)))


def find_loose_checkpoint_dirs(results_dir: Path, only: Optional[str] = None) -> List[Dict[str, Any]]:
    """Discover top-level `checkpoints_*/` dirs that have no experiment directory.

    `evaluate_agents.py` writes its weights straight to
    `results/checkpoints_evaluate_agents/` (and `checkpoints_ppo_evaluate_agents/`)
    with no summary JSON at all -- it never calls the summary-writing path
    `dqn_experiment.py`/`ppo_experiment.py` use. Those runs produce the README's
    headline matched-budget figures, so they have to be scoreable even though
    `find_runs` can't see them. Board size comes from `evaluate_agents.py`'s own
    ROWS/COLS/NUM_MINES constants, which are the 5x5/5-mine benchmark.
    """
    runs: List[Dict[str, Any]] = []
    if not results_dir.exists():
        return runs

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("checkpoints_"):
            continue
        if only and only not in child.name:
            continue
        runs.append(
            {
                "name": child.name,
                "agent": "ppo" if "ppo" in child.name else "dqn",
                "rows": DEFAULT_BOARD[0],
                "cols": DEFAULT_BOARD[1],
                "mines": DEFAULT_BOARD[2],
                "network_size": "default",
                "checkpoint_dir": child,
                "reported_win_rate": None,
                "reported_eval_episodes": None,
                # No summary exists, so there is no recorded deployment choice
                # to preserve -- both files are scored and reported as-is.
                "reported_used_checkpoint": None,
            }
        )
    return runs


def find_runs(results_dir: Path, only: Optional[str] = None) -> List[Dict[str, Any]]:
    """Discover every experiment directory holding a summary and a checkpoint dir.

    Mirrors `ResultsLoader._discover`'s "glob one level deep for
    `*_summary.json`" convention, but additionally requires a
    `checkpoints_*/` directory, since a run with no saved weights can't be
    re-scored.
    """
    runs: List[Dict[str, Any]] = []
    for summary_path in sorted(results_dir.glob("*/*_summary.json")):
        run_dir = summary_path.parent
        if only and only not in run_dir.name:
            continue
        checkpoint_dirs = sorted(run_dir.glob("checkpoints_*"))
        if not checkpoint_dirs:
            continue

        summary = json.loads(summary_path.read_text())
        agent = "ppo" if summary_path.name.startswith("ppo_") else "dqn"
        rows = summary.get("rows") or DEFAULT_BOARD[0]
        cols = summary.get("cols") or DEFAULT_BOARD[1]
        mines = summary.get("mines") or DEFAULT_BOARD[2]

        runs.append(
            {
                "name": run_dir.name,
                "agent": agent,
                "rows": rows,
                "cols": cols,
                "mines": mines,
                "network_size": summary.get("network_size", "default"),
                "checkpoint_dir": checkpoint_dirs[-1],
                "reported_win_rate": summary.get("win_rate"),
                "reported_eval_episodes": summary.get("eval_episodes"),
                "reported_used_checkpoint": summary.get("used_checkpoint"),
            }
        )
    return runs


def evaluate_checkpoint(run: Dict[str, Any], checkpoint_path: Path, episodes: int, eval_seed: int = EVAL_SEED) -> Dict[str, Any]:
    """Score one checkpoint file over `episodes` greedy episodes.

    Always evaluates under `reward_mode="default"` regardless of how the run
    was trained, for the same reason `ppo_experiment.py` does: a shaped-reward
    run's inflated per-step rewards would make `avg_reward` incomparable
    across experiments. Win rate is already reward-scale-invariant.
    """
    rows, cols, mines = run["rows"], run["cols"], run["mines"]

    if run["agent"] == "dqn":
        agent: Any = DQNAgent(rows=rows, cols=cols, network_size=run["network_size"], seed=0)
    else:
        agent = PPOAgent(rows=rows, cols=cols, seed=0)
    metadata = agent.load_checkpoint(checkpoint_path)

    env = MinesweeperEnv(rows=rows, cols=cols, num_mines=mines, seed=eval_seed, reward_mode="default")

    wins = 0
    total_steps = 0
    total_reward = 0.0
    for _ in range(episodes):
        observation, info = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.select_action(observation, explore=False)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
        if info.get("won", False):
            wins += 1

    low, high = wilson_interval(wins, episodes)
    return {
        "wins": wins,
        "episodes": episodes,
        "win_rate": wins / episodes,
        "win_rate_ci95": [round(low, 3), round(high, 3)],
        "avg_episode_length": total_steps / episodes,
        "avg_reward": total_reward / episodes,
        "checkpoint_metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the re-evaluation sweep."""
    parser = argparse.ArgumentParser(
        description="Re-score every saved checkpoint at a higher episode count than the original summaries used."
    )
    parser.add_argument(
        "--episodes", type=int, default=2000, help="Greedy evaluation episodes per checkpoint."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory holding the experiment subdirectories (needs checkpoints, so the full local results/).",
    )
    parser.add_argument(
        "--only", type=str, default=None, help="Only evaluate runs whose directory name contains this substring."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/reevaluation_report.json",
        help="Where to write the JSON report. Deliberately outside results_public/: the backend's "
        "ResultsLoader scans every *.json under that tree (and namespaces a lone JSON file by its "
        "parent directory), so any report written there surfaces as a bogus experiment on "
        "/api/experiments -- the same failure mode the loader excludes replays/ to avoid.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=EVAL_SEED,
        help="Seed for the evaluation boards. Held identical across every row in one report, so a "
        "difference between rows is a difference between models. Pass 42 to score on the same boards "
        "the experiment summaries use, making these numbers comparable to the published ones too.",
    )
    parser.add_argument("--torch-threads", type=int, default=None, help="Cap CPU threads PyTorch uses.")
    return parser.parse_args()


def main() -> None:
    """Re-score every discoverable checkpoint and write a single JSON report."""
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    results_dir = Path(args.results_dir)
    runs = find_runs(results_dir, args.only) + find_loose_checkpoint_dirs(results_dir, args.only)
    if not runs:
        raise SystemExit(f"No runs with checkpoints found under {results_dir}")

    print(f"Re-evaluating {len(runs)} runs at {args.episodes} episodes each (eval seed {args.eval_seed})\n")
    report: Dict[str, Any] = {
        "eval_episodes": args.episodes,
        "eval_seed": args.eval_seed,
        "eval_reward_mode": "default",
        "runs": {},
    }

    start = time.time()
    for run in runs:
        names = DQN_CHECKPOINTS if run["agent"] == "dqn" else PPO_CHECKPOINTS
        entry: Dict[str, Any] = {
            "agent": run["agent"],
            "board": f"{run['rows']}x{run['cols']}",
            "mines": run["mines"],
            "network_size": run["network_size"],
            "reported_win_rate": run["reported_win_rate"],
            "reported_eval_episodes": run["reported_eval_episodes"],
            "reported_used_checkpoint": run["reported_used_checkpoint"],
            "checkpoints": {},
        }
        for label, filename in names.items():
            path = run["checkpoint_dir"] / filename
            if not path.exists():
                continue
            result = evaluate_checkpoint(run, path, args.episodes, args.eval_seed)
            entry["checkpoints"][label] = result
            print(
                f"  {run['name']:<38} {label:<5} {result['wins']:>4}/{args.episodes} = "
                f"{result['win_rate'] * 100:5.2f}%  CI[{result['win_rate_ci95'][0]:.2f}, {result['win_rate_ci95'][1]:.2f}]"
            )
        report["runs"][run["name"]] = entry

    report["total_seconds"] = round(time.time() - start, 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {output_path} in {report['total_seconds']}s")


if __name__ == "__main__":
    main()
