"""Run the cumulative ablation that explains how the v2 DQN reached 77% .

Why this exists: `results/dqn_v2_A_baseline` (100k episodes, 77.25%) differs
from the last clean v1 ablation arm (`exp_F_masked_target`, 11.40%) in *eight*
ways at once -- network preset, no-guess training boards, first-click policy,
reward scale, replay ratio, epsilon schedule, episode budget, and checkpoint
selection. A single run moving eight variables cannot say which of them bought
the improvement, so the pipeline currently has to present the jump as an
unexplained leap. This script turns it into a ladder where each rung adds
exactly one factor, and the difference between two adjacent rungs *is* that
factor's contribution.

Two variables are deliberately held constant across the ladder rather than
swept inside it:

  * `--first-click-safe area` -- unlike every other option here, this one
    applies to *evaluation* as well as training, so it moves the benchmark
    rather than the agent. Sweeping it inside the ladder would make two rungs
    incomparable. It is instead pinned for arms A-G and isolated once, in arm
    H, whose delta against F is the honest answer to "how much of the headline
    is the easier board distribution?".
  * `--episodes 100000` -- so that a rung's delta is its factor rather than its
    factor plus a different training budget. The budget question gets its own
    arm (G) against F.

Checkpoint selection (`best_model.pt`) is left at the experiment script's
default on every arm, matching the run being explained; it is therefore
constant and cannot contaminate a delta.

Arm F re-runs the existing `dqn_v2_A_baseline` configuration under the family
naming scheme. That is not redundant: it shares `--seed 42`, so it should
reproduce 77.25% exactly, and a ladder whose top rung does not reproduce the
run it claims to explain is a ladder nobody should trust. Pass
`--skip-existing` to reuse a completed arm instead of retraining it.

Results land in `results/dqn_abl_<LETTER>_<variant>`, following the
`_<LETTER>_<variant>` convention `evaluation/dqn_experiment.py` uses for
backend-grouped families, so the whole ladder surfaces as one family
("dqn_abl") once mirrored into `results_public/`.

Run with:
    python -m evaluation.dqn_v2_ablation --dry-run
    python -m evaluation.dqn_v2_ablation --arms A B C D
    python -m evaluation.dqn_v2_ablation --skip-existing --torch-threads 4

Then report the ladder with the existing comparison script:
    python -m evaluation.compare_ablation --results-dir results \\
        --experiments dqn_abl_A_base dqn_abl_B_reward_scale ...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Held constant on every arm: board, seeds, and the replay/optimizer settings
# that were never in question. `--eval-seed` is pinned separately from `--seed`
# by `dqn_experiment` itself, so every arm is scored on the *same* 2,000
# boards and a delta between arms is a difference in play, not in draw.
FIXED_FLAGS: List[str] = [
    "--rows", "5",
    "--cols", "5",
    "--mines", "5",
    "--seed", "42",
    "--eval-seed", "42",
    "--lr", "1e-4",
    "--batch-size", "64",
    "--buffer-size", "20000",
    "--target-update-every", "25",
    "--epsilon-min", "0.05",
    "--checkpoint-every", "10000",
]

# Arm A's starting point: the pre-v2 configuration, but at the ladder's episode
# budget and benchmark so it is comparable to every rung above it. These are
# the defaults `exp_F_masked_target` ran under -- its summary predates the
# schema that records them, which is why they are spelled out here rather than
# read from that run.
BASE_CONFIG: Dict[str, str] = {
    "--episodes": "100000",
    "--first-click-safe": "area",
    "--network-size": "default",
    "--reward-scale": "1.0",
    "--train-every": "1",
    "--epsilon-decay": "0.995",
}


@dataclass(frozen=True)
class Arm:
    """One rung of the ladder.

    `overrides` are applied on top of the *previous* arm's resolved config for
    cumulative arms, or on top of a named arm's config for the standalone
    comparison arms (G, H) that branch off the top rung instead of extending
    it.
    """

    letter: str
    variant: str
    adds: str
    overrides: Dict[str, str] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    # None => extends the previous arm (cumulative). A letter => branches off
    # that arm, so it is a controlled comparison against it rather than a rung.
    branches_from: Optional[str] = None
    est_minutes: int = 25

    @property
    def name(self) -> str:
        return f"dqn_abl_{self.letter}_{self.variant}"


# The ladder. A-F are cumulative: each adds one factor to the arm above it, so
# `win_rate(N) - win_rate(N-1)` is that factor's contribution *at that point in
# the sequence*. G and H branch off F, isolating the two variables the ladder
# holds fixed.
ARMS: List[Arm] = [
    Arm(
        letter="A",
        variant="base",
        adds="pre-v2 configuration at the ladder's budget and benchmark",
        overrides=dict(BASE_CONFIG),
        # train-every 1 is four gradient steps per transition, so this arm and
        # B cost roughly four times what the rungs above them do.
        est_minutes=50,
    ),
    Arm(
        letter="B",
        variant="reward_scale",
        adds="--reward-scale 0.1 (keeps TD targets in smooth_l1's quadratic regime)",
        overrides={"--reward-scale": "0.1"},
        est_minutes=50,
    ),
    Arm(
        letter="C",
        variant="train_every",
        adds="--train-every 4 (standard DQN replay ratio)",
        overrides={"--train-every": "4"},
    ),
    Arm(
        letter="D",
        variant="epsilon",
        adds="--epsilon-decay 0.9997 (exploration lasts ~15k episodes, not ~600)",
        overrides={"--epsilon-decay": "0.9997"},
    ),
    Arm(
        letter="E",
        variant="fully_conv",
        adds="--network-size fully_conv (1x1 conv head, board-size independent)",
        overrides={"--network-size": "fully_conv"},
    ),
    Arm(
        letter="F",
        variant="solvable",
        adds="--guarantee-solvable (no-guess training boards; eval stays unfiltered)",
        flags=["--guarantee-solvable"],
    ),
    Arm(
        letter="G",
        variant="short",
        adds="F at 25k episodes -- was the 100k budget necessary?",
        overrides={"--episodes": "25000"},
        branches_from="F",
        est_minutes=7,
    ),
    Arm(
        letter="H",
        variant="unsafe_first_click",
        adds="F with --first-click-safe none -- how much of the result is the benchmark?",
        overrides={"--first-click-safe": "none"},
        branches_from="F",
    ),
]


def resolve_configs() -> Dict[str, List[str]]:
    """Walk the ladder, accumulating each arm's full flag list.

    Cumulative arms inherit the previous arm's resolved config; branching arms
    inherit the arm named in `branches_from`. Returns arm name -> CLI flags.
    """
    resolved: Dict[str, Dict[str, str]] = {}
    resolved_flags: Dict[str, List[str]] = {}
    configs: Dict[str, List[str]] = {}
    previous: Optional[str] = None

    for arm in ARMS:
        parent = arm.branches_from or previous
        if parent is None:
            config: Dict[str, str] = {}
            extra_flags: List[str] = []
        else:
            config = dict(resolved[parent])
            extra_flags = list(resolved_flags[parent])

        config.update(arm.overrides)
        extra_flags.extend(arm.flags)

        resolved[arm.letter] = config
        resolved_flags[arm.letter] = extra_flags

        cli: List[str] = list(FIXED_FLAGS)
        for key, value in config.items():
            cli.extend([key, value])
        cli.extend(extra_flags)
        cli.extend(["--output-dir", f"results/{arm.name}"])
        configs[arm.letter] = cli

        # Branching arms are dead ends: the next cumulative arm continues from
        # the last arm that was itself cumulative.
        if arm.branches_from is None:
            previous = arm.letter

    return configs


def build_command(flags: List[str], torch_threads: Optional[int]) -> List[str]:
    """Full `python -m evaluation.dqn_experiment ...` command for one arm."""
    command = [sys.executable, "-m", "evaluation.dqn_experiment", *flags]
    if torch_threads is not None:
        command.extend(["--torch-threads", str(torch_threads)])
    return command


def is_complete(arm: Arm, results_dir: Path) -> bool:
    """True if this arm already wrote a summary, i.e. finished a full run."""
    return any((results_dir / arm.name).glob("dqn_history_*_summary.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the cumulative DQN v2 ablation ladder (arms A-H)."
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        metavar="LETTER",
        help="Subset of arm letters to run, e.g. --arms A B C. Default: all. "
        "Note that a cumulative arm's delta is only meaningful once the arm "
        "below it has also been run.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Where arms write, used only to detect already-completed runs "
        "(--output-dir itself stays relative, as dqn_experiment expects).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an arm whose results directory already holds a summary, "
        "instead of retraining it.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Passed through to each run -- cap this when running arms in "
        "parallel from separate shells.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ladder and the exact command per arm without running.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = resolve_configs()

    selected = ARMS
    if args.arms:
        wanted = {letter.upper() for letter in args.arms}
        unknown = wanted - {arm.letter for arm in ARMS}
        if unknown:
            parser_choices = ", ".join(arm.letter for arm in ARMS)
            raise SystemExit(f"Unknown arm(s): {sorted(unknown)}; choose from {parser_choices}")
        selected = [arm for arm in ARMS if arm.letter in wanted]

    total_estimate = sum(arm.est_minutes for arm in selected)
    print(f"DQN v2 ablation -- {len(selected)} arm(s), ~{total_estimate} min estimated\n")

    failures: List[str] = []
    for arm in selected:
        relation = f"branches from {arm.branches_from}" if arm.branches_from else "cumulative"
        print(f"[{arm.letter}] {arm.name}  ({relation}, ~{arm.est_minutes} min)")
        print(f"     adds: {arm.adds}")

        command = build_command(configs[arm.letter], args.torch_threads)

        if args.dry_run:
            print(f"     {' '.join(command)}\n")
            continue

        if args.skip_existing and is_complete(arm, args.results_dir):
            print("     already complete, skipping\n")
            continue

        started = time.time()
        result = subprocess.run(command)
        elapsed = (time.time() - started) / 60

        if result.returncode != 0:
            # Keep going: a later arm may still be runnable, and a partial
            # ladder is more useful than none. Failures are re-reported at the
            # end so they cannot scroll past unnoticed.
            print(f"     FAILED (exit {result.returncode}) after {elapsed:.1f} min\n")
            failures.append(arm.name)
        else:
            print(f"     done in {elapsed:.1f} min\n")

    if failures:
        print(f"{len(failures)} arm(s) failed: {', '.join(failures)}")
        raise SystemExit(1)

    if not args.dry_run:
        names = " ".join(arm.name for arm in selected)
        print("Report the ladder with:")
        print(f"    python -m evaluation.compare_ablation --results-dir results --experiments {names}")


if __name__ == "__main__":
    main()
