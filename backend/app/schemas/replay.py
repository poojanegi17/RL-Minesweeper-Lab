"""Schemas for replay listing/detail endpoints (`GET /api/replays[/{id}]`).

Every field here is declared explicitly (no `extra="allow"`), which is a
deliberate security property, not just Pydantic's default: FastAPI's
`response_model` re-serializes through these schemas, so any key present in
a replay JSON file that *isn't* declared below -- including, hypothetically,
a hand-edited or buggy file that somehow contained mine positions -- is
silently dropped rather than passed through to the API response. See
`services/replay_loader.py` and the README's Replay Visualization section
for the full security picture (the primary guarantee is that mine positions
are never written to a replay file in the first place; this is the backstop).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ReplayAction(BaseModel):
    row: int
    col: int


class ReplayStep(BaseModel):
    """One recorded timestep. `board_state` is the observation *after* this
    action -- exactly what the agent would see next, never anything further
    ahead. `reasoning` is agent-specific and optional; `None` when nothing
    was recorded for that agent type (see `rl/evaluation/replay.py`)."""

    step: int
    board_state: List[List[int]]
    action: ReplayAction
    reward: float
    done: bool
    reasoning: Optional[Dict[str, Any]] = None


class ReplaySummary(BaseModel):
    """One row of `GET /api/replays`."""

    id: str = Field(..., description="Filename stem, e.g. \"dqn_episode_143\".")
    agent: str
    experiment_id: Optional[str] = Field(
        None, description="Which experiment's checkpoint the agent was loaded from, if applicable (DQN/PPO only)."
    )
    won: bool
    steps: int = Field(..., description="Total number of steps the episode took.")


class ReplayDetail(BaseModel):
    """Full timeline for `GET /api/replays/{id}`."""

    id: str
    agent: str
    experiment_id: Optional[str] = None
    board_size: str
    mines: int
    seed: Optional[int] = None
    episode_number: Optional[int] = None
    generated_at: Optional[str] = None
    initial_board: List[List[int]] = Field(..., description="The all-hidden board before any action.")
    timeline: List[ReplayStep]
    won: bool
    total_reward: float
    steps: int = Field(..., description="Total number of steps the episode took (== len(timeline)).")
