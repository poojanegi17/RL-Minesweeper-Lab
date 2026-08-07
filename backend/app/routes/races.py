"""`GET /api/races` and `GET /api/races/{id}`."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.race import RaceDetail, RaceSummary
from app.services.race_loader import MalformedRaceError, RaceLoader, RaceNotFoundError, get_race_loader

router = APIRouter(prefix="/api/races", tags=["races"])


def to_summary(raw: Dict[str, Any]) -> RaceSummary:
    return RaceSummary(
        id=raw["id"],
        seed=raw["seed"],
        board_size=raw["board_size"],
        mines=raw["mines"],
        agents=list(raw["agents"].keys()),
        generated_at=raw.get("generated_at"),
    )


def to_detail(raw: Dict[str, Any]) -> RaceDetail:
    return RaceDetail(
        id=raw["id"],
        seed=raw["seed"],
        board_size=raw["board_size"],
        mines=raw["mines"],
        generated_at=raw.get("generated_at"),
        initial_board=raw["initial_board"],
        agents=raw["agents"],
    )


@router.get("", response_model=List[RaceSummary])
def list_races(loader: RaceLoader = Depends(get_race_loader)) -> List[RaceSummary]:
    """List every race discoverable under rl/results/races/. Never errors -- an
    empty or missing directory simply yields an empty list."""
    return [to_summary(raw) for raw in loader.list_races()]


@router.get("/{race_id}", response_model=RaceDetail)
def get_race(race_id: str, loader: RaceLoader = Depends(get_race_loader)) -> RaceDetail:
    """Full per-agent timelines for one race."""
    try:
        raw = loader.get_race(race_id)
    except RaceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MalformedRaceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return to_detail(raw)
