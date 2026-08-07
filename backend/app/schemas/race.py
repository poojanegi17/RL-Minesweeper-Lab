"""Schemas for race listing/detail endpoints (`GET /api/races[/{id}]`).

A "race" is several agents' full timelines recorded on the *same* seed --
see `rl/evaluation/generate_race.py` -- so every agent in one race bundle
played the identical mine layout. Reuses `ReplayStep`/`ReplayAction` from
`schemas/replay.py` rather than redeclaring an identical shape: a race
agent's per-step timeline is exactly a replay's timeline, just nested one
level deeper under the agent's name instead of being its own top-level
resource.

Every field is declared explicitly (no `extra="allow"`), same security
property as `schemas/replay.py`: a race file can never leak mine positions
through this API even by accident, since `FastAPI`'s `response_model`
re-serializes strictly through these fields.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.replay import ReplayStep


class RaceAgentResult(BaseModel):
    """One agent's full episode within a race -- everything `ReplayDetail` has
    except `initial_board` (identical across every agent in the race, so it's
    stored once on `RaceDetail` instead of duplicated per agent)."""

    experiment_id: Optional[str] = Field(
        None, description="Which experiment's checkpoint this agent was loaded from, if applicable (DQN/PPO only)."
    )
    steps: List[ReplayStep]
    won: bool
    total_reward: float
    steps_taken: int = Field(..., description="Total number of steps this agent's episode took.")


class RaceSummary(BaseModel):
    """One row of `GET /api/races`."""

    id: str = Field(..., description="Filename stem, e.g. \"race_42\".")
    seed: int
    board_size: str
    mines: int
    agents: List[str] = Field(..., description="Display names of the agents recorded in this race, e.g. [\"Random\", \"CSP\", \"DQN\", \"PPO\"].")
    generated_at: Optional[str] = None


class RaceDetail(BaseModel):
    """Full detail for `GET /api/races/{id}`."""

    id: str
    seed: int
    board_size: str
    mines: int
    generated_at: Optional[str] = None
    initial_board: List[List[int]] = Field(..., description="The all-hidden board before any agent acted -- identical for every agent in this race.")
    agents: Dict[str, RaceAgentResult]
