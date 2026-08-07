"""The single source of truth for every board size/mine-density preset this
project offers, beyond the original fixed 5x5/5-mine benchmark.

Every script that generates level/density-scoped data (`evaluate_board_config.py`,
`generate_replays.py`, `generate_race.py`) and the backend's `/api/board-configs`
endpoint import from here, so the level/density catalog only ever needs to
change in one place.

Density presets keep mine density in a roughly similar band across board
sizes (~12-30%). "standard" density at the "beginner" level is deliberately
the existing 5-mine board -- unchanged from every experiment this project
has run so far, so nothing about the current 5x5 data needs to move or be
regenerated to fit this catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class LevelConfig:
    """One board size and its mine-density presets."""

    rows: int
    cols: int
    densities: Dict[str, int]  # density name -> mine count


BOARD_LEVELS: Dict[str, LevelConfig] = {
    "beginner": LevelConfig(rows=5, cols=5, densities={"sparse": 3, "standard": 5, "dense": 8}),
    "intermediate": LevelConfig(rows=9, cols=9, densities={"sparse": 8, "standard": 12, "dense": 18}),
    "expert": LevelConfig(rows=16, cols=16, densities={"sparse": 30, "standard": 40, "dense": 60}),
}

LEVEL_ORDER = ["beginner", "intermediate", "expert"]
DENSITY_ORDER = ["sparse", "standard", "dense"]


def resolve(level: str, density: str) -> tuple[int, int, int]:
    """Return `(rows, cols, mines)` for a `(level, density)` pair, or raise
    `KeyError` with a clear message if either isn't in the catalog."""
    if level not in BOARD_LEVELS:
        raise KeyError(f"Unknown level {level!r}; choose from {LEVEL_ORDER}")
    level_config = BOARD_LEVELS[level]
    if density not in level_config.densities:
        raise KeyError(f"Unknown density {density!r} for level {level!r}; choose from {DENSITY_ORDER}")
    return level_config.rows, level_config.cols, level_config.densities[density]
