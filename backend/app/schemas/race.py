"""Schemas for race listing/detail endpoints (`GET /api/races[/{id}]`).

A "race" is several agents taking turns on one *physically shared* board --
see `rl/evaluation/shared_race.py` for why that's a meaningfully different
thing from several independent episodes matched by seed. Reuses
`ReplayAction` from `schemas/replay.py` for a turn's action rather than
redeclaring an identical `{row, col}` shape.

Every field is declared explicitly (no `extra="allow"`), same security
property as `schemas/replay.py`: an eliminated agent's fatal cell is only
ever recoverable from that turn's own `action` + `eliminated` flag (exactly
the same "derivable, never separately marked" convention `ReplayDetail`
already uses for a lost episode's fatal move) -- it is never written into
`board_state` at all, at generation time, for any turn, so there's nothing
for this API to accidentally leak even if it wanted to.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.replay import ReplayAction


class RaceTurn(BaseModel):
    """One turn: which agent acted, what they chose, and the *shared* board
    immediately after -- `eliminated` is true when this turn's action was a
    mine (in which case `board_state` still shows that cell as hidden; see
    module docstring)."""

    turn: int
    agent: str
    action: ReplayAction
    board_state: List[List[int]]
    eliminated: bool
    reasoning: Optional[Dict[str, Any]] = None


class RaceSummary(BaseModel):
    """One row of `GET /api/races`."""

    id: str = Field(..., description="Filename stem, e.g. \"race_42\".")
    seed: int
    board_size: str
    mines: int
    turn_order: List[str] = Field(..., description="Fixed round-robin turn order, e.g. [\"Random\", \"CSP\", \"DQN\", \"PPO\"].")
    won: bool = Field(..., description="Whether the board was collectively cleared before every agent was eliminated.")
    total_turns: int
    generated_at: Optional[str] = None


class RaceDetail(BaseModel):
    """Full detail for `GET /api/races/{id}`."""

    id: str
    seed: int
    board_size: str
    mines: int
    turn_order: List[str]
    generated_at: Optional[str] = None
    initial_board: List[List[int]] = Field(..., description="The all-hidden board before any agent acted.")
    turns: List[RaceTurn]
    won: bool
    total_turns: int
    surviving_agents: List[str]
    eliminated_agents: Dict[str, int] = Field(..., description="Agent name -> the turn number that eliminated them.")
