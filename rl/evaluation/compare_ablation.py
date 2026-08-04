"""Compare a set of DQN stabilization experiments (e.g. baseline vs. checkpoint
selection vs. LR decay vs. smaller network), each run into its own results
subdirectory via `evaluation.dqn_experiment --output-dir <subdir>`.

"Best WR" is the win rate of the best checkpoint seen during training (from
periodic evaluation, if the run recorded one); "Deployed WR" is the win rate
of whatever was actually evaluated at the end (the best checkpoint if
`--no-best-checkpoint` wasn't passed to the experiment, otherwise the raw
final weights). Loss statistics are computed over the second half of
training, since that's where instability has shown up in prior runs.

Run with:
    python -m evaluation.compare_ablation --results-dir results \\
        --experiments exp_A_baseline exp_B_checkpoint exp_C_lr_decay exp_D_small_net
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_summary(experiment_dir: Path) -> Optional[Dict[str, Any]]:
    summary_paths = list(experiment_dir.glob("dqn_history_*_summary.json"))
    return json.loads(summary_paths[0].read_text()) if summary_paths else None


def _load_history(experiment_dir: Path) -> Optional[List[Dict[str, Any]]]:
    history_paths = [
        p for p in experiment_dir.glob("dqn_history_*.json") if not p.name.endswith("_summary.json")
    ]
    return json.loads(history_paths[0].read_text()) if history_paths else None


def _stability_stats(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Loss mean/max/stdev over the second half of training episodes."""
    half = len(history) // 2
    losses = [h["loss"] for h in history[half:] if h.get("loss") is not None]
    if not losses:
        return {"loss_mean": float("nan"), "loss_max": float("nan"), "loss_stdev": float("nan")}
    return {
        "loss_mean": statistics.mean(losses),
        "loss_max": max(losses),
        "loss_stdev": statistics.pstdev(losses) if len(losses) > 1 else 0.0,
    }


def main() -> None:
    """Parse args, load each experiment's summary/history, and print a comparison table."""
    parser = argparse.ArgumentParser(description="Compare DQN stabilization experiments.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--experiments", nargs="+", required=True, help="Subdirectory names under --results-dir, one per experiment."
    )
    args = parser.parse_args()

    root = Path(args.results_dir)
    print(
        f"{'Experiment':<22} {'Best WR':>8} {'Deployed WR':>12} {'Avg Reward':>11} "
        f"{'Loss Mean':>10} {'Loss Max':>10} {'Loss Stdev':>11}"
    )
    for name in args.experiments:
        experiment_dir = root / name
        summary = _load_summary(experiment_dir)
        history = _load_history(experiment_dir)
        if summary is None or history is None:
            print(f"{name:<22} (missing results in {experiment_dir})")
            continue

        best_checkpoint = summary.get("best_checkpoint_metadata") or {}
        best_wr = best_checkpoint.get("win_rate")
        stability = _stability_stats(history)

        best_wr_str = f"{best_wr * 100:>7.1f}%" if best_wr is not None else "     n/a"
        print(
            f"{name:<22} "
            f"{best_wr_str} "
            f"{summary['win_rate'] * 100:>11.1f}% "
            f"{summary['avg_reward']:>11.2f} "
            f"{stability['loss_mean']:>10.2f} "
            f"{stability['loss_max']:>10.2f} "
            f"{stability['loss_stdev']:>11.2f}"
        )


if __name__ == "__main__":
    main()
