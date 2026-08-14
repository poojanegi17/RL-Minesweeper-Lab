"""Subsample the per-episode histories under `results_public/` so they are small
enough to commit.

Why this exists
---------------
A 100,000-episode run writes a ~40 MB history JSON. There are dozens of them,
which is over a gigabyte of artifacts for a repository whose entire point is
that the numbers are committed and reproducible. Git never forgets, so every
re-run of an experiment would add another 40 MB to history permanently -- the
repository would grow without bound while the *site* never used the extra
resolution for anything.

It never used it because the frontend already downsamples before rendering
(`frontend/src/lib/downsample.ts`): drawing 100,000 SVG points is slower than
drawing 2,000 and looks identical. So the full-resolution file was being
shipped, parsed, transferred, and then thrown away in the browser.

What this does and does not do
------------------------------
It **subsamples**: it keeps a subset of the real recorded rows and deletes the
rest. It does not average, smooth, bucket, or synthesise anything. Every row in
a compacted file is a row the training loop actually wrote, with its original
`episode` number intact -- so a chart's x-axis still runs to 100,000 and every
plotted point is a real episode, just fewer of them.

That distinction matters for this project specifically: an averaged row would
be a datapoint that never happened, sitting in a file that claims to be a
recording. A missing row is only lower resolution.

The first and last rows are always kept, so the start and end of every curve
are exact rather than whatever the sampling stride happened to land on.

The full-resolution originals stay in `rl/results/`, which is gitignored --
nothing is destroyed, and re-running an experiment regenerates them.

Idempotent: a file already at or below the target is left untouched, so this is
safe to run after every mirroring step.

Run with:
    python -m evaluation.compact_public_histories --dry-run
    python -m evaluation.compact_public_histories
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

# Chart-resolution target. The frontend downsamples to a few hundred points for
# rendering, so 2,000 leaves comfortable headroom for zooming or a future denser
# chart while still being ~50x smaller than a 100,000-episode file.
DEFAULT_MAX_ROWS = 2000

# Only these are subsampled. Summaries hold the published figures and are tiny;
# board results, replays and races are already small and are each a *whole*
# artifact rather than a sampled series -- dropping rows from a replay would
# delete moves from a recorded game.
HISTORY_GLOB = "**/*_history_*.json"


def subsample(rows: List[Dict[str, Any]], max_rows: int) -> List[Dict[str, Any]]:
    """Evenly-spaced subset of `rows`, always including the first and last.

    Mirrors `frontend/src/lib/downsample.ts`'s stride so a compacted file and a
    browser-side downsample of the original pick comparable points, with the
    endpoints pinned so curves start and end where the run actually did.
    """
    if len(rows) <= max_rows or max_rows <= 0:
        return rows

    step = len(rows) / max_rows
    picked = [rows[int(i * step)] for i in range(max_rows)]
    if picked[-1] is not rows[-1]:
        picked[-1] = rows[-1]
    return picked


def compact_file(path: Path, max_rows: int, dry_run: bool) -> tuple[int, int, int, int]:
    """Returns (rows_before, rows_after, bytes_before, bytes_after)."""
    before_bytes = path.stat().st_size
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  ! skipping {path} -- unreadable: {exc}")
        return (0, 0, before_bytes, before_bytes)

    if not isinstance(rows, list):
        return (0, 0, before_bytes, before_bytes)

    kept = subsample(rows, max_rows)
    if len(kept) == len(rows):
        return (len(rows), len(rows), before_bytes, before_bytes)

    # Compact separators: this file is machine-read by the API, never hand-edited,
    # and the pretty-printing was a meaningful share of its size.
    payload = json.dumps(kept, separators=(",", ":"))
    if not dry_run:
        path.write_text(payload)
    return (len(rows), len(kept), before_bytes, len(payload.encode()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Subsample committed training histories so they fit in git.")
    parser.add_argument("--results-dir", type=str, default="results_public", help="Tree to compact in place.")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS, help="Rows to keep per history file.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change and exit.")
    args = parser.parse_args()

    root = Path(args.results_dir)
    if not root.is_dir():
        raise SystemExit(f"No such directory: {root}")

    files = sorted(p for p in root.glob(HISTORY_GLOB) if not p.name.endswith("_summary.json"))
    if not files:
        print(f"No history files found under {root}")
        return

    total_before = total_after = 0
    changed = 0
    for path in files:
        rows_before, rows_after, b_before, b_after = compact_file(path, args.max_rows, args.dry_run)
        total_before += b_before
        total_after += b_after
        if rows_after and rows_after != rows_before:
            changed += 1
            print(f"  {path.relative_to(root)}: {rows_before:,} -> {rows_after:,} rows, "
                  f"{b_before / 1024**2:.1f} MB -> {b_after / 1024**2:.2f} MB")

    verb = "would shrink" if args.dry_run else "shrank"
    print(
        f"\n{changed} of {len(files)} files {verb}: "
        f"{total_before / 1024**3:.2f} GB -> {total_after / 1024**3:.3f} GB "
        f"({100 * (1 - total_after / total_before):.1f}% smaller)"
    )
    if args.dry_run:
        print("Dry run -- nothing written.")


if __name__ == "__main__":
    main()
