"""Write `reevaluate_checkpoints.py`'s figures back into the experiment summaries.

Why this is a separate script
-----------------------------
`reevaluate_checkpoints.py` is read-only: it scores checkpoints and writes one
report. This script is the destructive counterpart -- it edits summary JSON in
place, so it's deliberately kept apart and requires an explicit `--write`.

What gets replaced, and with which number
-----------------------------------------
Each summary's evaluation block is replaced with the 2,000-episode score **of
the checkpoint that experiment actually deployed**, read from its own
`used_checkpoint` field. That distinction matters: the ablation's whole point is
that some runs reported `final_model.pt` and others `best_model.pt`, and those
two differ by more than any training variable tested. Substituting the wrong
one would silently rewrite what the experiment *was*.

Provenance is not lost. The original 200-episode figures stay recorded in
`analysis/reevaluation_report.json` (`reported_win_rate` /
`reported_eval_episodes`), and every summary this script touches gains an
`evaluation_source` marker so a reader can tell the number no longer came from
the training script that wrote the rest of the file.

Run with:
    python -m evaluation.apply_reevaluation                 # dry run, prints a diff
    python -m evaluation.apply_reevaluation --write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EVALUATION_SOURCE = "reevaluate_checkpoints.py"

# `used_checkpoint` values written by dqn_experiment.py / ppo_experiment.py,
# mapped to which of the two scored checkpoints that run actually deployed.
_DEPLOYED = {
    "best_model.pt": "best",
    "best_policy.pt": "best",
    "final_in_memory_weights": "final",
}


def resolve_deployed(used_checkpoint: Optional[str], available: Dict[str, Any]) -> Optional[str]:
    """Pick which scored checkpoint corresponds to what the run deployed.

    Returns None when the summary records a `used_checkpoint` this script
    doesn't recognize, so an unexpected value is skipped loudly rather than
    silently defaulting to the wrong weights.
    """
    if used_checkpoint is None:
        return None
    label = _DEPLOYED.get(used_checkpoint)
    if label is None or label not in available:
        return None
    return label


def build_evaluation_block(scored: Dict[str, Any], episodes: int) -> Dict[str, Any]:
    """The evaluation fields to merge into a summary, from one scored checkpoint."""
    return {
        "win_rate": scored["win_rate"],
        "win_rate_ci95": scored["win_rate_ci95"],
        "avg_reward": round(scored["avg_reward"], 4),
        "avg_episode_length": scored["avg_episode_length"],
        "failures": episodes - scored["wins"],
        "eval_episodes": episodes,
        "evaluation_source": EVALUATION_SOURCE,
    }


def find_summary(run_dir: Path) -> Optional[Path]:
    """The single `*_summary.json` inside an experiment directory, if present."""
    matches = sorted(run_dir.glob("*_summary.json"))
    return matches[0] if len(matches) == 1 else None


def plan_updates(report: Dict[str, Any], results_dirs: List[Path]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Work out every (summary file, new evaluation block) pair to write.

    Returns `(updates, skipped)` -- `skipped` holds human-readable reasons, so a
    run that can't be matched up is reported rather than passed over in silence.
    """
    episodes = report["eval_episodes"]
    updates: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for name, entry in sorted(report["runs"].items()):
        label = resolve_deployed(entry.get("reported_used_checkpoint"), entry["checkpoints"])
        if label is None:
            skipped.append(f"{name}: unrecognized used_checkpoint {entry.get('reported_used_checkpoint')!r}")
            continue

        block = build_evaluation_block(entry["checkpoints"][label], episodes)
        found_any = False
        for results_dir in results_dirs:
            summary_path = find_summary(results_dir / name)
            if summary_path is None:
                continue
            found_any = True
            updates.append(
                {
                    "run": name,
                    "path": summary_path,
                    "deployed": label,
                    "old_win_rate": entry.get("reported_win_rate"),
                    "old_eval_episodes": entry.get("reported_eval_episodes"),
                    "block": block,
                }
            )
        if not found_any:
            skipped.append(f"{name}: no summary JSON found under any results dir")

    return updates, skipped


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Apply reevaluate_checkpoints.py's figures to the experiment summaries."
    )
    parser.add_argument(
        "--report", type=str, default="analysis/reevaluation_report.json", help="Report to apply."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        action="append",
        default=None,
        help="Results directory to patch; repeatable. Defaults to results_public/ and results/.",
    )
    parser.add_argument(
        "--write", action="store_true", help="Actually modify files. Without this, prints a dry-run diff."
    )
    return parser.parse_args()


def main() -> None:
    """Print, and optionally apply, the summary updates implied by the report."""
    args = parse_args()
    report = json.loads(Path(args.report).read_text())

    candidates = args.results_dir or ["results_public", "results"]
    results_dirs = [Path(d) for d in candidates if Path(d).exists()]
    if not results_dirs:
        raise SystemExit(f"None of these results directories exist: {candidates}")

    updates, skipped = plan_updates(report, results_dirs)

    print(f"{'Applying' if args.write else 'Dry run --'} {len(updates)} summary updates "
          f"at {report['eval_episodes']} episodes\n")
    for update in updates:
        old = update["old_win_rate"]
        old_text = "n/a" if old is None else f"{old * 100:.2f}%"
        ci = update["block"]["win_rate_ci95"]
        print(
            f"  {update['run']:<40} [{update['deployed']:<5}] "
            f"{old_text:>7} @{update['old_eval_episodes'] or '?':<5} -> "
            f"{update['block']['win_rate'] * 100:5.2f}% @{update['block']['eval_episodes']} "
            f"CI[{ci[0]:.2f}, {ci[1]:.2f}]   {update['path']}"
        )

    if skipped:
        print("\nSkipped:")
        for reason in skipped:
            print(f"  {reason}")

    if not args.write:
        print("\nNothing written. Re-run with --write to apply.")
        return

    for update in updates:
        summary = json.loads(update["path"].read_text())
        summary.update(update["block"])
        update["path"].write_text(json.dumps(summary, indent=2))

    print(f"\nWrote {len(updates)} summary files.")


if __name__ == "__main__":
    main()
