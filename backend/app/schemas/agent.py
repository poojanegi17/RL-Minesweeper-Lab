"""Schemas for the static agent catalog (`GET /api/agents`).

Unlike experiments, agents are not derived from `rl/results/` -- Random,
CSP, and Q-Learning never write experiment artifacts at all (see
`services/results_loader.py`'s module docstring), so this is fixed project
metadata, not a filesystem read.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentInfo(BaseModel):
    """One entry in the agent catalog."""

    name: str = Field(..., description="Agent name, e.g. \"DQN\".")
    type: str = Field(..., description="Category, e.g. \"deep_rl\" or \"constraint_solver\".")
    description: str = Field(..., description="One-line human-readable description.")
    has_experiment_artifacts: bool = Field(
        ..., description="Whether this agent has any experiment files under rl/results/ to query via /api/experiments."
    )
