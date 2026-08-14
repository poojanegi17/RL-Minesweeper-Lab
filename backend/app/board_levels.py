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


# The two board distributions this project has published results under, and
# the results subtree each one's grid lives in. `first_click_safe="area"`
# keeps the whole 3x3 block around the opening click mine-free (v2); "none"
# places mines before the first click, so the opening move can lose (v1).
# They are different games -- a win rate under one is not comparable to a win
# rate under the other -- which is why they are separate trees rather than a
# flag on one set of files.
FIRST_CLICK_POLICY_DIRS: Dict[str, str] = {"area": "v2", "none": "v1"}


def is_valid_first_click_policy(policy: str) -> bool:
    return policy in FIRST_CLICK_POLICY_DIRS


def resolve_policy_level_dir(base_results_dir: Path, policy: str, level: str, density: str) -> Path:
    """Which directory a `(policy, level, density)` selection reads from.

    Unlike `resolve_level_dir`, "beginner"/"standard" is *not* aliased to the
    root results directory here: the versioned trees carry their own
    `levels/beginner/standard/` and the root holds only whichever distribution
    happened to be re-baselined into it last. Aliasing would silently serve
    that root under both policies and make the two indistinguishable, which is
    the one thing this resolver exists to prevent.
    """
    return base_results_dir / FIRST_CLICK_POLICY_DIRS[policy] / "levels" / level / density
