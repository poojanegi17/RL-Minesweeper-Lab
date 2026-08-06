"""Reads and validates replay artifacts under `rl/results/replays/`.

Mirrors `results_loader.py`'s shape (discovery, resilient listing vs.
strict single-item fetch, dependency-injected for tests) but is
deliberately a separate service rather than folded into `ResultsLoader`:
replays are a different kind of artifact with much simpler discovery (one
flat directory of self-contained JSON files, no summary/history pairing).

Written by `rl/evaluation/generate_replays.py` -- one file per episode,
`{agent}_episode_{n}.json`. No filesystem writes happen here; this module
only reads.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings

logger = logging.getLogger(__name__)

# Fields `rl/evaluation/replay.py`'s `build_replay()` always writes -- a
# replay missing any of these is treated as malformed rather than silently
# rendered with gaps, since these are exactly what the viewer needs to be
# meaningful at all.
REQUIRED_REPLAY_KEYS = {"id", "agent", "initial_board", "steps", "won", "total_reward", "steps_taken"}


class ReplayLoaderError(Exception):
    """Base class for all replay-loading errors, so routes can catch one type."""


class ReplayNotFoundError(ReplayLoaderError):
    """Raised when a replay id doesn't match any discovered file."""


class MalformedReplayError(ReplayLoaderError):
    """Raised when a replay file exists but isn't valid/expected JSON."""


class ReplayLoader:
    """Discovers and reads replay artifacts under a `rl/results/replays/`-shaped directory."""

    def __init__(self, replays_dir: Path) -> None:
        self.replays_dir = Path(replays_dir)

    def _discover(self) -> Dict[str, Path]:
        """Map replay id (filename stem) -> path. Empty (not an error) if the
        directory doesn't exist -- no replays generated yet is a valid state."""
        if not self.replays_dir.exists():
            logger.warning("replays_dir %s does not exist; treating as empty", self.replays_dir)
            return {}
        try:
            return {path.stem: path for path in sorted(self.replays_dir.glob("*.json"))}
        except OSError as exc:
            logger.warning("could not list %s: %s", self.replays_dir, exc)
            return {}

    def _load(self, path: Path) -> Dict[str, Any]:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise MalformedReplayError(f"Could not parse replay JSON at {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise MalformedReplayError(f"Expected a JSON object in {path}, got {type(data).__name__}")
        missing = REQUIRED_REPLAY_KEYS - data.keys()
        if missing:
            raise MalformedReplayError(f"Replay {path} is missing required fields: {sorted(missing)}")
        return data

    def list_replays(self) -> List[Dict[str, Any]]:
        """Return every discoverable replay's raw dict, skipping (and logging) unparseable ones.

        A single malformed replay must not take down the whole listing --
        same resilience contract as `ResultsLoader.list_experiments`.
        """
        records: List[Dict[str, Any]] = []
        for replay_id, path in sorted(self._discover().items()):
            try:
                records.append(self._load(path))
            except MalformedReplayError as exc:
                logger.warning("skipping replay %r in listing: %s", replay_id, exc)
        return records

    def get_replay(self, replay_id: str) -> Dict[str, Any]:
        """Return one replay's raw dict, or raise if it doesn't exist / can't be parsed."""
        path = self._discover().get(replay_id)
        if path is None:
            raise ReplayNotFoundError(f"No replay found with id {replay_id!r}")
        return self._load(path)


def get_replay_loader() -> ReplayLoader:
    """FastAPI dependency provider. Tests override this via `app.dependency_overrides`
    to point at a temp directory instead of the real `rl/results/replays/`."""
    return ReplayLoader(settings.results_dir / "replays")
