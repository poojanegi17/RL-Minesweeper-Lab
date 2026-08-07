"""`GET /api/board-configs` -- the level/density catalog, so the frontend
builds its toggle/dropdown from real data instead of a hardcoded copy."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter

from app.board_levels import BOARD_LEVELS, LEVEL_ORDER
from app.schemas.board_config import BoardLevelInfo

router = APIRouter(prefix="/api/board-configs", tags=["board-configs"])


@router.get("", response_model=List[BoardLevelInfo])
def list_board_configs() -> List[BoardLevelInfo]:
    """Every difficulty level and its mine-density presets, in display order."""
    return [
        BoardLevelInfo(level=level, rows=BOARD_LEVELS[level].rows, cols=BOARD_LEVELS[level].cols, densities=BOARD_LEVELS[level].densities)
        for level in LEVEL_ORDER
    ]
