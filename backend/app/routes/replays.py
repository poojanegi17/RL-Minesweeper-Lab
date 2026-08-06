"""`GET /api/replays` and `GET /api/replays/{id}`."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

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


@router.get("", response_model=List[ReplaySummary])
def list_replays(loader: ReplayLoader = Depends(get_replay_loader)) -> List[ReplaySummary]:
    """List every replay discoverable under rl/results/replays/. Never errors -- an
    empty or missing directory simply yields an empty list."""
    return [to_summary(raw) for raw in loader.list_replays()]


@router.get("/{replay_id}", response_model=ReplayDetail)
def get_replay(replay_id: str, loader: ReplayLoader = Depends(get_replay_loader)) -> ReplayDetail:
    """Full step-by-step timeline for one replay."""
    try:
        raw = loader.get_replay(replay_id)
    except ReplayNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MalformedReplayError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return to_detail(raw)
