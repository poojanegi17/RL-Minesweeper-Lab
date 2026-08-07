"""Reads `{agent}_board_result.json` files written by
`rl/evaluation/evaluate_board_config.py` -- one per agent evaluated at a
given (level, density), outside the "beginner"/"standard" default (which
still reads through `ResultsLoader`'s experiment-artifact machinery,
unchanged).

Deliberately a separate, much simpler loader rather than folded into
`ResultsLoader`: a board result is never a training "experiment" (no
history, no hyperparameters) -- most of these runs involve no training at
all (CSP/Random need no checkpoint at any size; DQN/PPO are evaluated, not
trained, at every density beyond a level's own "standard" run). Forcing this
shape through `ResultsLoader`'s history+summary discovery would misrepresent
what these files actually are.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

REQUIRED_BOARD_RESULT_KEYS = {"agent", "level", "density", "rows", "cols", "mines", "win_rate", "avg_episode_length", "avg_reward"}


class BoardResultLoaderError(Exception):
    """Base class for all board-result-loading errors."""


class MalformedBoardResultError(BoardResultLoaderError):
    """Raised when a board-result file exists but isn't valid/expected JSON."""


class BoardResultLoader:
    """Discovers and reads `{agent}_board_result.json` files under one
    `(level, density)` directory (see `app.board_levels.resolve_level_dir`)."""

    def __init__(self, level_density_dir: Path) -> None:
        self.level_density_dir = Path(level_density_dir)

    def list_results(self) -> List[Dict[str, Any]]:
        """Return every discoverable board result's raw dict, skipping (and
        logging) unparseable ones. Empty (not an error) if the directory
        doesn't exist -- no evaluation run yet at this level/density is a
        valid, expected state (e.g. DQN/PPO before that level is trained)."""
        if not self.level_density_dir.exists():
            return []
        results: List[Dict[str, Any]] = []
        try:
            paths = sorted(self.level_density_dir.glob("*_board_result.json"))
        except OSError as exc:
            logger.warning("could not list %s: %s", self.level_density_dir, exc)
            return []

        for path in paths:
            try:
                data = json.loads(path.read_text())
                if not isinstance(data, dict):
                    raise MalformedBoardResultError(f"Expected a JSON object in {path}, got {type(data).__name__}")
                missing = REQUIRED_BOARD_RESULT_KEYS - data.keys()
                if missing:
                    raise MalformedBoardResultError(f"Board result {path} is missing required fields: {sorted(missing)}")
                results.append(data)
            except (OSError, json.JSONDecodeError, MalformedBoardResultError) as exc:
                logger.warning("skipping board result %r: %s", path, exc)
        return results
