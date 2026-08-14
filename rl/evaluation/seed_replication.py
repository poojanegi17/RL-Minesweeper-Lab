"""Re-run DQN Beginner's post-fix variants across several training seeds.

Every DQN figure on the Research page is a single training run. The only
evidence about run-to-run variance the project has is accidental: `exp_C_lr_decay`
and `exp_E_combined` were configured identically and scored 1.60% and 1.25%, a
0.35pp spread. That number is doing a lot of work -- it is the caveat the whole
Beginner level is qualified by -- and it rests on one pair of runs.

This script replaces it with a measurement. It re-runs each of the five post-fix
variants at additional training seeds, holding the evaluation boards fixed
(`--eval-seed`, see `dqn_experiment.py`), so the spread it reports is training
variance alone rather than training variance confounded with a different test
set. The question it answers is specific: the F->G gap the level currently calls
null (11.40% vs 12.85%, p = 0.16) is 4x the 0.35pp spread, so is that spread an
underestimate, an overestimate, or about right?

The pre-fix variants are deliberately not included. The bootstrap mask is a code
path in `DQNAgent._double_dqn_targets`, not a CLI flag, so reproducing them would
mean reverting the fix -- and their 21x deficit is far outside anything seed
noise could explain.

Runs land in `results_public/seed_replication/<variant>_s<seed>/`, one directory
deeper than `ResultsLoader._discover` scans. That is intentional: these are a
methodological control, not new pipeline cards, and the API would otherwise
surface each one as its own ablation variant.

Run with:
    python -m evaluation.seed_replication run
    python -m evaluation.seed_replication run --seeds 43 44 45 --workers 4
    python -m evaluation.seed_replication report
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

RL_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = RL_DIR / "results_public"
REPLICATION_DIR = RESULTS_DIR / "seed_replication"

EPISODES = 25000
# The seed every existing run used, and the fixed evaluation board sequence.
ORIGINAL_SEED = 42

# Each post-fix variant, as the exact flags that reproduce the archived run --
# read back off its own summary JSON, not from memory. `--no-best-checkpoint`
# and `--checkpoint-every 2500` are common to all five (every archived summary
# has used_checkpoint="final_in_memory_weights" and checkpoint_every=2500), so
# they live in COMMON_FLAGS rather than being repeated per variant.
COMMON_FLAGS = ["--no-best-checkpoint", "--checkpoint-every", "2500"]

VARIANTS: Dict[str, Dict[str, object]] = {
    "masked_target": {"original": "exp_F_masked_target", "flags": []},
    "masked_lr_decay": {
        "original": "exp_G_masked_lr_decay",
        "flags": ["--lr-schedule", "0:1e-4,10000:5e-5,20000:1e-5"],
    },
    "masked_shaped": {"original": "exp_H_masked_shaped", "flags": ["--reward-mode", "shaped"]},
    "masked_slow_epsilon": {
        "original": "exp_I_masked_slow_epsilon",
        "flags": ["--epsilon-decay", "0.9998"],
    },
    "masked_deep": {"original": "exp_J_masked_deep", "flags": ["--network-size", "deep"]},
}


def run_dir(variant: str, seed: int) -> Path:
    """Where a single (variant, seed) run writes its history and summary."""
    return REPLICATION_DIR / f"{variant}_s{seed}"


def summary_path(directory: Path) -> Path:
    """The summary JSON `dqn_experiment` writes into `directory`."""
    return directory / f"dqn_history_{EPISODES}_summary.json"


def build_command(variant: str, seed: int, torch_threads: Optional[int]) -> List[str]:
    """Assemble the `dqn_experiment` invocation for one (variant, seed) run."""
    flags = list(VARIANTS[variant]["flags"])  # type: ignore[arg-type]
    command = [
        sys.executable,
        "-m",
        "evaluation.dqn_experiment",
        "--episodes",
        str(EPISODES),
        "--seed",
        str(seed),
        "--eval-seed",
        str(ORIGINAL_SEED),
        "--output-dir",
        str(run_dir(variant, seed)),
        *COMMON_FLAGS,
        *flags,
    ]
    if torch_threads is not None:
        command += ["--torch-threads", str(torch_threads)]
    return command


def execute(variant: str, seed: int, torch_threads: Optional[int], overwrite: bool) -> str:
    """Run one (variant, seed) training run, skipping it if already complete."""
    directory = run_dir(variant, seed)
    if summary_path(directory).exists() and not overwrite:
        return f"skip  {variant} seed={seed} (already complete)"

    start = time.time()
    result = subprocess.run(
        build_command(variant, seed, torch_threads),
        cwd=RL_DIR,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        return f"FAIL  {variant} seed={seed} after {elapsed:.0f}s\n      " + "\n      ".join(tail)
    return f"done  {variant} seed={seed} in {elapsed / 60:.1f} min -- {result.stdout.strip().splitlines()[-1]}"


def command_run(args: argparse.Namespace) -> None:
    """Launch the (variant x seed) grid, `--workers` runs at a time."""
    REPLICATION_DIR.mkdir(parents=True, exist_ok=True)
    variants = args.variants or list(VARIANTS)
    jobs = [(variant, seed) for variant in variants for seed in args.seeds]

    print(
        f"{len(jobs)} runs ({len(variants)} variants x {len(args.seeds)} seeds), "
        f"{args.workers} at a time, {EPISODES} episodes each.\n"
        f"Evaluation boards held fixed at seed {ORIGINAL_SEED} for every run.\n"
        f"Writing to {REPLICATION_DIR}\n",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(execute, variant, seed, args.torch_threads, args.overwrite): (variant, seed)
            for variant, seed in jobs
        }
        for future in as_completed(futures):
            print(future.result(), flush=True)

    print("\nAll runs finished. Aggregate with: python -m evaluation.seed_replication report")


def load_win_rates(variant: str, seeds: Sequence[int]) -> Dict[int, float]:
    """Collect this variant's win rate per seed, original run included."""
    rates: Dict[int, float] = {}

    original = RESULTS_DIR / str(VARIANTS[variant]["original"]) / f"dqn_history_{EPISODES}_summary.json"
    if original.exists():
        rates[ORIGINAL_SEED] = json.loads(original.read_text())["win_rate"]

    for seed in seeds:
        path = summary_path(run_dir(variant, seed))
        if path.exists():
            rates[seed] = json.loads(path.read_text())["win_rate"]
    return rates


def two_proportion_p(rate_a: float, n_a: int, rate_b: float, n_b: int) -> float:
    """Two-sided p-value for two independent proportions, normal approximation."""
    successes = rate_a * n_a + rate_b * n_b
    pooled = successes / (n_a + n_b)
    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if standard_error == 0:
        return 1.0
    z = abs(rate_a - rate_b) / standard_error
    return math.erfc(z / math.sqrt(2))


def command_report(args: argparse.Namespace) -> None:
    """Aggregate every completed run into per-variant spread and comparisons."""
    eval_episodes = 2000
    table: Dict[str, Dict[int, float]] = {
        variant: load_win_rates(variant, args.seeds) for variant in VARIANTS
    }

    print(f"Per-variant win rate by training seed ({eval_episodes} fixed evaluation boards each)\n")
    header = "variant".ljust(22) + "".join(f"seed {s}".rjust(10) for s in [ORIGINAL_SEED, *args.seeds])
    print(header + "      mean      sd     range")
    print("-" * len(header + "      mean      sd     range"))

    for variant, rates in table.items():
        cells = "".join(
            (f"{rates[s] * 100:9.2f}%" if s in rates else "         -")
            for s in [ORIGINAL_SEED, *args.seeds]
        )
        values = [v * 100 for v in rates.values()]
        if len(values) >= 2:
            mean = statistics.mean(values)
            sd = statistics.stdev(values)
            spread = max(values) - min(values)
            print(f"{variant.ljust(22)}{cells}{mean:9.2f}%{sd:8.2f}{spread:9.2f}pp")
        else:
            print(f"{variant.ljust(22)}{cells}        -       -         -   (1 run)")

    complete = {v: r for v, r in table.items() if len(r) >= 2}
    if not complete:
        print("\nNo variant has 2+ seeds yet -- run `seed_replication run` first.")
        return

    all_spreads = [max(r.values()) - min(r.values()) for r in complete.values()]
    pooled_sd = statistics.mean(
        [statistics.stdev([v * 100 for v in r.values()]) for r in complete.values()]
    )
    print(
        f"\nSeed-to-seed spread across {len(complete)} replicated variants: "
        f"mean sd {pooled_sd:.2f}pp, widest range {max(all_spreads) * 100:.2f}pp."
    )
    print(
        "The project's existing caveat -- derived from one accidental replicate pair -- puts this "
        "at 0.35pp. Compare against it before reading any variant difference as real."
    )

    print("\nVariant means vs masked_target, pooled across seeds:")
    if "masked_target" not in complete:
        print("  masked_target has no replicates yet; skipping.")
        return
    reference = complete["masked_target"]
    reference_mean = statistics.mean(reference.values())
    reference_n = eval_episodes * len(reference)
    for variant, rates in complete.items():
        if variant == "masked_target":
            continue
        mean = statistics.mean(rates.values())
        p = two_proportion_p(mean, eval_episodes * len(rates), reference_mean, reference_n)
        verdict = "significant" if p < 0.05 else "not significant"
        print(
            f"  {variant.ljust(22)} {mean * 100:6.2f}% vs {reference_mean * 100:6.2f}%   "
            f"p = {p:.3g}  ({verdict}, pooled)"
        )
    print(
        "\nPooled p-values treat every episode as independent, so they understate uncertainty when\n"
        "the seed is the real unit of replication. Read them alongside the sd column above: a gap\n"
        "smaller than the seed spread is not a result, whatever the pooled p-value says."
    )


def parse_args() -> argparse.Namespace:
    """Parse the run/report subcommand and its options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Train the (variant x seed) grid.")
    run_parser.add_argument(
        "--seeds", type=int, nargs="+", default=[43, 44], help="Training seeds to add (42 already exists)."
    )
    run_parser.add_argument(
        "--variants", type=str, nargs="+", choices=list(VARIANTS), default=None, help="Subset of variants."
    )
    run_parser.add_argument("--workers", type=int, default=4, help="Concurrent training runs.")
    run_parser.add_argument(
        "--torch-threads",
        type=int,
        default=3,
        help="CPU threads per run. Keep workers x threads at or under the core count.",
    )
    run_parser.add_argument("--overwrite", action="store_true", help="Re-run runs that already completed.")
    run_parser.set_defaults(func=command_run)

    report_parser = subparsers.add_parser("report", help="Aggregate completed runs.")
    report_parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44], help="Replication seeds to look for.")
    report_parser.set_defaults(func=command_report)

    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
