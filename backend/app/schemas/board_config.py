"""Schema for `GET /api/board-configs`."""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class BoardLevelInfo(BaseModel):
    """One difficulty level and its mine-density presets."""

    level: str = Field(..., description="e.g. \"beginner\", \"intermediate\", \"expert\".")
    rows: int
    cols: int
    densities: Dict[str, int] = Field(..., description="Density name -> mine count, e.g. {\"sparse\": 3, \"standard\": 5, \"dense\": 8}.")
