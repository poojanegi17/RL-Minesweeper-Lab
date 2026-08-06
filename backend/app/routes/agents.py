"""`GET /api/agents` -- the static agent catalog.

Not filesystem-derived: see `services/results_loader.py`'s module docstring
for why Random/CSP/Q-Learning have no experiment artifacts to read.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter

from app.schemas.agent import AgentInfo

router = APIRouter(prefix="/api/agents", tags=["agents"])

_AGENT_CATALOG: List[AgentInfo] = [
    AgentInfo(
        name="Random",
        type="baseline",
        description="Picks a uniformly random hidden cell each step; establishes the performance floor.",
        has_experiment_artifacts=False,
    ),
    AgentInfo(
        name="CSP",
        type="constraint_solver",
        description="Constraint-satisfaction based solver using logical deduction over revealed numbers.",
        has_experiment_artifacts=False,
    ),
    AgentInfo(
        name="Q-Learning",
        type="tabular_rl",
        description="Tabular Q-learning over exact board patterns; doesn't generalize across large boards.",
        has_experiment_artifacts=False,
    ),
    AgentInfo(
        name="DQN",
        type="deep_rl",
        description="CNN-based Double DQN with experience replay, a target network, and checkpoint selection.",
        has_experiment_artifacts=True,
    ),
    AgentInfo(
        name="PPO",
        type="deep_rl",
        description="CNN actor-critic trained with GAE and a clipped surrogate objective; supports reward shaping.",
        has_experiment_artifacts=True,
    ),
]


@router.get("", response_model=List[AgentInfo])
def list_agents() -> List[AgentInfo]:
    """Return the fixed catalog of agents implemented in this project."""
    return _AGENT_CATALOG
