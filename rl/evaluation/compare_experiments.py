"""Print a comparison table across DQN training-budget experiments.

Reads the summary JSON files produced by `evaluation.dqn_experiment` (one per
`--episodes` value it was run with) and prints a table sorted by episode count.

Run with:
    python -m evaluation.compare_experiments --results-dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Parse args, load summaries, and print a comparison table."""
    parser = argparse.ArgumentParser(description="Compare DQN experiments by training budget.")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    summary_paths = sorted(
        Path(args.results_dir).glob("dqn_history_*_summary.json"),
        key=lambda p: json.loads(p.read_text())["episodes"],
    )
    if not summary_paths:
        print(f"No summary files found in {args.results_dir}")
        return

    print(f"{'Episodes':>10} {'Win Rate':>10} {'Avg Length':>12} {'Failures':>10} {'Train Time':>12}")
    for path in summary_paths:
        summary = json.loads(path.read_text())
        print(
            f"{summary['episodes']:>10} "
            f"{summary['win_rate'] * 100:>9.1f}% "
            f"{summary['avg_episode_length']:>12.2f} "
            f"{summary['failures']:>10} "
            f"{summary['train_seconds']:>10.1f}s"
        )


if __name__ == "__main__":
    main()
