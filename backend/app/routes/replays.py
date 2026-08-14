"""`GET /api/replays` and `GET /api/replays/{id}`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.board_levels import (
    DEFAULT_DENSITY,
    DEFAULT_LEVEL,
    is_valid,
    is_valid_first_click_policy,
    resolve_level_dir,
    resolve_policy_level_dir,
)
from app.schemas.replay import ReplayDetail, ReplaySummary
from app.services.replay_loader import MalformedReplayError, ReplayLoader, ReplayNotFoundError, get_replay_loader

router = APIRouter(prefix="/api/replays", tags=["replays"])


def to_summary(raw: Dict[str, Any]) -> ReplaySummary:
    return ReplaySummary(
        id=raw["id"],
        agent=raw["agent"],
        experiment_id=raw.get("experiment_id"),
        won=raw["won"],
        steps=raw["steps_taken"],
    )


def to_detail(raw: Dict[str, Any]) -> ReplayDetail:
    return ReplayDetail(
        id=raw["id"],
        agent=raw["agent"],
        experiment_id=raw.get("experiment_id"),
        board_size=raw.get("board_size", ""),
        mines=raw.get("mines", 0),
        seed=raw.get("seed"),
        episode_number=raw.get("episode_number"),
        generated_at=raw.get("generated_at"),
        initial_board=raw["initial_board"],
        timeline=raw["steps"],
        won=raw["won"],
        total_reward=raw["total_reward"],
        steps=raw["steps_taken"],
    )


def _scoped_loader(
    level: str, density: str, loader: ReplayLoader, first_click_safe: Optional[str] = None
) -> ReplayLoader:
    """`loader.replays_dir` is always `{results_dir}/replays` (see
    `get_replay_loader`, including in tests' dependency overrides), so
    `.parent` reliably recovers the base results directory to resolve a
    different level/density against -- no need to reach into `app.config`
    directly, and level-scoped tests can keep using the same
    dependency-override pattern as every other test."""
    if first_click_safe is not None:
        # An explicit policy always reads that distribution's own tree. The
        # two are different games, so a replay recorded under one must never
        # be served as an episode from the other.
        scoped = resolve_policy_level_dir(loader.replays_dir.parent, first_click_safe, level, density)
        return ReplayLoader(scoped / "replays")
    if level == DEFAULT_LEVEL and density == DEFAULT_DENSITY:
        return loader
    scoped_dir = resolve_level_dir(loader.replays_dir.parent, level, density)
    return ReplayLoader(scoped_dir / "replays")


@router.get("", response_model=List[ReplaySummary])
def list_replays(
    level: str = Query(DEFAULT_LEVEL, description="Difficulty level, e.g. \"beginner\"."),
    density: str = Query(DEFAULT_DENSITY, description="Mine-density preset, e.g. \"standard\"."),
    first_click_safe: Optional[str] = Query(
        None,
        description='Board distribution: "area" (opening click opens a mine-free 3x3 block) or "none". '
        "Omit for the default tree.",
    ),
    loader: ReplayLoader = Depends(get_replay_loader),
) -> List[ReplaySummary]:
    """List every replay discoverable at the given `(level, density)` (see
    `GET /api/board-configs`). Never errors -- an empty or missing directory
    (e.g. an agent not yet trained at this level) simply yields an empty list."""
    if not is_valid(level, density):
        raise HTTPException(status_code=400, detail=f"Unknown level/density: {level!r}/{density!r}")
    if first_click_safe is not None and not is_valid_first_click_policy(first_click_safe):
        raise HTTPException(status_code=400, detail=f"Unknown first_click_safe policy: {first_click_safe!r}")
    return [to_summary(raw) for raw in _scoped_loader(level, density, loader, first_click_safe).list_replays()]


@router.get("/{replay_id}", response_model=ReplayDetail)
def get_replay(
    replay_id: str,
    level: str = Query(DEFAULT_LEVEL, description="Difficulty level, e.g. \"beginner\"."),
    density: str = Query(DEFAULT_DENSITY, description="Mine-density preset, e.g. \"standard\"."),
    first_click_safe: Optional[str] = Query(
        None,
        description='Board distribution: "area" (opening click opens a mine-free 3x3 block) or "none". '
        "Omit for the default tree.",
    ),
    loader: ReplayLoader = Depends(get_replay_loader),
) -> ReplayDetail:
    """Full step-by-step timeline for one replay at the given `(level, density)`."""
    if not is_valid(level, density):
        raise HTTPException(status_code=400, detail=f"Unknown level/density: {level!r}/{density!r}")
    try:
        raw = _scoped_loader(level, density, loader, first_click_safe).get_replay(replay_id)
    except ReplayNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MalformedReplayError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return to_detail(raw)
