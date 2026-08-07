"""`GET /api/races` and `GET /api/races/{id}`."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query

from app.board_levels import DEFAULT_DENSITY, DEFAULT_LEVEL, is_valid, resolve_level_dir
from app.schemas.race import RaceDetail, RaceSummary
from app.services.race_loader import MalformedRaceError, RaceLoader, RaceNotFoundError, get_race_loader

router = APIRouter(prefix="/api/races", tags=["races"])


def to_summary(raw: Dict[str, Any]) -> RaceSummary:
    return RaceSummary(
        id=raw["id"],
        seed=raw["seed"],
        board_size=raw["board_size"],
        mines=raw["mines"],
        turn_order=raw["turn_order"],
        won=raw["won"],
        total_turns=raw["total_turns"],
        generated_at=raw.get("generated_at"),
    )


def to_detail(raw: Dict[str, Any]) -> RaceDetail:
    return RaceDetail(
        id=raw["id"],
        seed=raw["seed"],
        board_size=raw["board_size"],
        mines=raw["mines"],
        turn_order=raw["turn_order"],
        generated_at=raw.get("generated_at"),
        initial_board=raw["initial_board"],
        turns=raw["turns"],
        won=raw["won"],
        total_turns=raw["total_turns"],
        surviving_agents=raw["surviving_agents"],
        eliminated_agents=raw["eliminated_agents"],
    )


def _scoped_loader(level: str, density: str, loader: RaceLoader) -> RaceLoader:
    """See `routes/replays.py`'s `_scoped_loader` -- same reasoning:
    `loader.races_dir.parent` reliably recovers the base results directory."""
    if level == DEFAULT_LEVEL and density == DEFAULT_DENSITY:
        return loader
    scoped_dir = resolve_level_dir(loader.races_dir.parent, level, density)
    return RaceLoader(scoped_dir / "races")


@router.get("", response_model=List[RaceSummary])
def list_races(
    level: str = Query(DEFAULT_LEVEL, description="Difficulty level, e.g. \"beginner\"."),
    density: str = Query(DEFAULT_DENSITY, description="Mine-density preset, e.g. \"standard\"."),
    loader: RaceLoader = Depends(get_race_loader),
) -> List[RaceSummary]:
    """List every race discoverable at the given `(level, density)` (see
    `GET /api/board-configs`). Never errors -- an empty or missing directory
    simply yields an empty list."""
    if not is_valid(level, density):
        raise HTTPException(status_code=400, detail=f"Unknown level/density: {level!r}/{density!r}")
    return [to_summary(raw) for raw in _scoped_loader(level, density, loader).list_races()]


@router.get("/{race_id}", response_model=RaceDetail)
def get_race(
    race_id: str,
    level: str = Query(DEFAULT_LEVEL, description="Difficulty level, e.g. \"beginner\"."),
    density: str = Query(DEFAULT_DENSITY, description="Mine-density preset, e.g. \"standard\"."),
    loader: RaceLoader = Depends(get_race_loader),
) -> RaceDetail:
    """Full turn-by-turn timeline for one shared-board race at the given `(level, density)`."""
    if not is_valid(level, density):
        raise HTTPException(status_code=400, detail=f"Unknown level/density: {level!r}/{density!r}")
    try:
        raw = _scoped_loader(level, density, loader).get_race(race_id)
    except RaceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MalformedRaceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return to_detail(raw)
