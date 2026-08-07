"""Reads and validates race artifacts under `rl/results/races/`.

Mirrors `replay_loader.py`'s shape (discovery, resilient listing vs. strict
single-item fetch, dependency-injected for tests) -- a race is a different
kind of artifact (several agents bundled per file, sharing one seed and one
initial board) but the discovery/reliability contract is identical, so it's
worth keeping the same pattern rather than inventing a new one.

Written by `rl/evaluation/generate_race.py` -- one file per shared seed,
`race_{seed}.json`. No filesystem writes happen here; this module only reads.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings

logger = logging.getLogger(__name__)

# Fields `rl/evaluation/replay.py`'s `build_shared_race()` always writes -- a
# race file missing any of these is treated as malformed rather than
# silently rendered with gaps.
REQUIRED_RACE_KEYS = {"id", "seed", "board_size", "mines", "turn_order", "initial_board", "turns", "won", "total_turns", "surviving_agents", "eliminated_agents"}


class RaceLoaderError(Exception):
    """Base class for all race-loading errors, so routes can catch one type."""


class RaceNotFoundError(RaceLoaderError):
    """Raised when a race id doesn't match any discovered file."""


class MalformedRaceError(RaceLoaderError):
    """Raised when a race file exists but isn't valid/expected JSON."""


class RaceLoader:
    """Discovers and reads race artifacts under a `rl/results/races/`-shaped directory."""

    def __init__(self, races_dir: Path) -> None:
        self.races_dir = Path(races_dir)

    def _discover(self) -> Dict[str, Path]:
        """Map race id (filename stem) -> path. Empty (not an error) if the
        directory doesn't exist -- no races generated yet is a valid state."""
        if not self.races_dir.exists():
            logger.warning("races_dir %s does not exist; treating as empty", self.races_dir)
            return {}
        try:
            return {path.stem: path for path in sorted(self.races_dir.glob("*.json"))}
        except OSError as exc:
            logger.warning("could not list %s: %s", self.races_dir, exc)
            return {}

    def _load(self, path: Path) -> Dict[str, Any]:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise MalformedRaceError(f"Could not parse race JSON at {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise MalformedRaceError(f"Expected a JSON object in {path}, got {type(data).__name__}")
        missing = REQUIRED_RACE_KEYS - data.keys()
        if missing:
            raise MalformedRaceError(f"Race {path} is missing required fields: {sorted(missing)}")
        return data

    def list_races(self) -> List[Dict[str, Any]]:
        """Return every discoverable race's raw dict, skipping (and logging) unparseable ones.

        A single malformed race must not take down the whole listing --
        same resilience contract as `ReplayLoader.list_replays`.
        """
        records: List[Dict[str, Any]] = []
        for race_id, path in sorted(self._discover().items()):
            try:
                records.append(self._load(path))
            except MalformedRaceError as exc:
                logger.warning("skipping race %r in listing: %s", race_id, exc)
        return records

    def get_race(self, race_id: str) -> Dict[str, Any]:
        """Return one race's raw dict, or raise if it doesn't exist / can't be parsed."""
        path = self._discover().get(race_id)
        if path is None:
            raise RaceNotFoundError(f"No race found with id {race_id!r}")
        return self._load(path)


def get_race_loader() -> RaceLoader:
    """FastAPI dependency provider. Tests override this via `app.dependency_overrides`
    to point at a temp directory instead of the real `rl/results/races/`."""
    return RaceLoader(settings.results_dir / "races")
