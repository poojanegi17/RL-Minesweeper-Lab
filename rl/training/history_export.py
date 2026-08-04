"""Export per-episode training history (as returned by an agent's `train()`) to disk.

Generic over history shape: any agent's `train()` returns a list of per-episode
dicts (as `QLearningAgent` and `DQNAgent` both already do), and this module just
serializes that list, numbering episodes along the way. It has no dependency on
Minesweeper or on any particular agent.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

PathLike = Union[str, Path]


def _with_episode_numbers(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a copy of `history` with a 1-indexed "episode" field prepended to each record."""
    return [{"episode": index + 1, **entry} for index, entry in enumerate(history)]


def save_history_json(history: List[Dict[str, Any]], path: PathLike) -> None:
    """Write per-episode training history to a JSON file as a list of records."""
    records = _with_episode_numbers(history)
    Path(path).write_text(json.dumps(records, indent=2))


def save_history_csv(history: List[Dict[str, Any]], path: PathLike) -> None:
    """Write per-episode training history to a CSV file, one row per episode.

    An empty `history` still produces an (empty) file, so callers don't need
    to special-case a training run that recorded no episodes.
    """
    records = _with_episode_numbers(history)
    path = Path(path)

    if not records:
        path.write_text("")
        return

    fieldnames = list(records[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
