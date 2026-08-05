"""Compare a set of PPO improvement experiments (e.g. baseline vs. longer
training vs. reward shaping vs. reward shaping + checkpoint selection), each
run into its own results subdirectory via `evaluation.ppo_experiment
--output-dir <subdir>`. Mirrors `evaluation.compare_ablation`'s shape for the
DQN stabilization experiments.

"Best WR" is the win rate of the best checkpoint seen during training (from
periodic evaluation, if the run recorded one); "Deployed WR" is the win rate
of whatever was actually evaluated at the end (the best checkpoint if
`--no-best-checkpoint` wasn't passed to the experiment, otherwise the raw
final weights). Loss and entropy statistics are computed over the second
half of training, matching `compare_ablation`'s convention, since that's
where any late-training instability would show up.

Run with:
    python -m evaluation.compare_ppo_experiments --results-dir results \\
        --experiments ppo_exp_A_baseline ppo_exp_B_longer ppo_exp_C_shaped ppo_exp_D_shaped_checkpoint
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_summary(experiment_dir: Path) -> Optional[Dict[str, Any]]:
    summary_paths = list(experiment_dir.glob("ppo_history_*_summary.json"))
    return json.loads(summary_paths[0].read_text()) if summary_paths else None


def _load_history(experiment_dir: Path) -> Optional[List[Dict[str, Any]]]:
    history_paths = [
        p for p in experiment_dir.glob("ppo_history_*.json") if not p.name.endswith("_summary.json")
    ]
    return json.loads(history_paths[0].read_text()) if history_paths else None


def _second_half_stats(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Policy/value loss mean and entropy mean/first/last over the second half of training."""
    half = len(history) // 2
    second_half = history[half:]

    def _mean(key: str) -> float:
        values = [h[key] for h in second_half if h.get(key) is not None]
        return statistics.mean(values) if values else float("nan")

    entropies = [h["entropy"] for h in history if h.get("entropy") is not None]
    return {
        "policy_loss_mean": _mean("policy_loss"),
        "value_loss_mean": _mean("value_loss"),
        "total_loss_mean": _mean("total_loss"),
        "entropy_mean_2nd_half": _mean("entropy"),
        "entropy_first": entropies[0] if entropies else float("nan"),
        "entropy_last": entropies[-1] if entropies else float("nan"),
    }


def main() -> None:
    """Parse args, load each experiment's summary/history, and print a comparison table."""
    parser = argparse.ArgumentParser(description="Compare PPO improvement experiments.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--experiments", nargs="+", required=True, help="Subdirectory names under --results-dir, one per experiment."
    )
    args = parser.parse_args()

    root = Path(args.results_dir)
    header = (
        f"{'Experiment':<28} {'Episodes':>8} {'Reward':>7} {'Best WR':>8} {'Deployed WR':>12} "
        f"{'Avg Reward':>11} {'Avg Len':>8} {'Entropy0':>9} {'EntropyN':>9} "
        f"{'PolicyLoss':>11} {'ValueLoss':>10}"
    )
    print(header)
    for name in args.experiments:
        experiment_dir = root / name
        summary = _load_summary(experiment_dir)
        history = _load_history(experiment_dir)
        if summary is None or history is None:
            print(f"{name:<28} (missing results in {experiment_dir})")
            continue

        best_checkpoint = summary.get("best_checkpoint_metadata") or {}
        best_wr = best_checkpoint.get("win_rate")
        stats = _second_half_stats(history)

        best_wr_str = f"{best_wr * 100:>7.1f}%" if best_wr is not None else "     n/a"
        print(
            f"{name:<28} "
            f"{summary['episodes']:>8} "
            f"{summary.get('reward_mode', 'default'):>7} "
            f"{best_wr_str} "
            f"{summary['win_rate'] * 100:>11.1f}% "
            f"{summary['avg_reward']:>11.2f} "
            f"{summary['avg_episode_length']:>8.2f} "
            f"{stats['entropy_first']:>9.3f} "
            f"{stats['entropy_last']:>9.3f} "
            f"{stats['policy_loss_mean']:>11.4f} "
            f"{stats['value_loss_mean']:>10.2f}"
        )


if __name__ == "__main__":
    main()
