"""Schemas for training-history metrics (`GET /api/experiments/{id}/metrics`) and the leaderboard."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MetricsResponse(BaseModel):
    """Per-episode training history for one experiment, chart-ready.

    `episodes` is row-oriented (one dict per episode, matching the source
    JSON almost exactly -- the shape most charting libraries, including
    Recharts, consume directly). `series` lists which metric keys are
    actually present for this agent type (DQN and PPO track different
    metrics), so a consumer doesn't have to guess or probe.
    """

    experiment_id: str
    agent: str
    total_episodes: int
    series: List[str] = Field(..., description="Metric keys present in every row of `episodes`.")
    episodes: List[Dict[str, Any]]


class LeaderboardEntry(BaseModel):
    """One row of `GET /api/leaderboard`."""

    agent: str
    win_rate: Optional[float] = None
    avg_episode_length: Optional[float] = None
    avg_reward: Optional[float] = None
    source: str = Field(
        ..., description="\"experiment_artifact\" (read live from rl/results/) or \"static_reference\" (README-recorded, no artifact exists)."
    )
    experiment_id: Optional[str] = Field(None, description="Which experiment this row's numbers came from, if artifact-sourced.")
