"""The backend's copy of the level/density catalog defined in `rl/board_configs.py`.

Duplicated rather than imported: the backend and `rl/` are separate deployable
services with separate dependencies (same reason `BENCHMARK_BOARD_ROWS` in
`app/config.py` is already an independent constant, not an import from `rl/`).
Keep the two in sync by hand if the catalog ever changes.

"beginner"/"standard" is a deliberate alias for the *root* results directory,
not a real `levels/beginner/standard/` subdirectory -- that's where every
experiment, replay, and race this project has ever generated already lives,
and duplicating ~270 files into a `levels/` subtree just to make the resolver
uniform would be pure waste. Every other (level, density) combination reads
from `results_dir/levels/{level}/{density}/`, populated by
`rl/evaluation/evaluate_board_config.py` / `generate_replays.py` /
`generate_race.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel


class LevelConfig(BaseModel):
    rows: int
    cols: int
    densities: Dict[str, int]  # density name -> mine count


BOARD_LEVELS: Dict[str, LevelConfig] = {
    "beginner": LevelConfig(rows=5, cols=5, densities={"sparse": 3, "standard": 5, "dense": 8}),
    "intermediate": LevelConfig(rows=9, cols=9, densities={"sparse": 8, "standard": 12, "dense": 18}),
    "expert": LevelConfig(rows=16, cols=16, densities={"sparse": 30, "standard": 40, "dense": 60}),
}

LEVEL_ORDER: List[str] = ["beginner", "intermediate", "expert"]
DENSITY_ORDER: List[str] = ["sparse", "standard", "dense"]

DEFAULT_LEVEL = "beginner"
DEFAULT_DENSITY = "standard"


def is_valid(level: str, density: str) -> bool:
    return level in BOARD_LEVELS and density in BOARD_LEVELS[level].densities


def resolve_level_dir(base_results_dir: Path, level: str, density: str) -> Path:
    """Which directory a `(level, density)` selection reads from. Callers
    should validate with `is_valid` first -- this does not raise for an
    unknown level/density, it just resolves to a (likely nonexistent)
    subdirectory, since every loader already treats a missing directory as
    "no data" rather than an error."""
    if level == DEFAULT_LEVEL and density == DEFAULT_DENSITY:
        return base_results_dir
    return base_results_dir / "levels" / level / density
