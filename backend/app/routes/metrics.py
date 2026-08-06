"""`GET /api/experiments/{id}/metrics` and `GET /api/leaderboard`.

Both are aggregate/derived views over experiment data rather than the raw
catalog or detail records `routes/experiments.py` and `routes/agents.py`
expose, hence living together here.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.metrics import LeaderboardEntry, MetricsResponse
from app.services.results_loader import (
    ExperimentNotFoundError,
    ExperimentRecord,
    MalformedArtifactError,
    ResultsLoader,
    get_results_loader,
)

router = APIRouter(prefix="/api", tags=["metrics"])

# Random, CSP, and Q-Learning write no experiment artifacts at all (see
# services/results_loader.py), so there is nothing under rl/results/ for
# these three to be measured from live. These are the project README's
# last-recorded `evaluation.evaluate_agents` figures, included so the
# leaderboard covers the full agent catalog -- but explicitly tagged
# `source="static_reference"` below rather than presented as if freshly
# read from a file, since they aren't.
_STATIC_REFERENCE_ENTRIES: List[LeaderboardEntry] = [
    LeaderboardEntry(agent="CSP", win_rate=0.455, avg_episode_length=6.67, source="static_reference"),
    LeaderboardEntry(agent="Random", win_rate=0.005, avg_episode_length=3.65, source="static_reference"),
    LeaderboardEntry(agent="Q-Learning", win_rate=0.005, avg_episode_length=4.06, source="static_reference"),
]


@router.get("/experiments/{experiment_id}/metrics", response_model=MetricsResponse, tags=["experiments"])
def get_experiment_metrics(experiment_id: str, loader: ResultsLoader = Depends(get_results_loader)) -> MetricsResponse:
    """Full per-episode training history for one experiment, chart-ready.

    `episodes` is row-oriented (matches Recharts' expected input shape
    directly); `series` lists which metric keys are present, since DQN and
    PPO track different diagnostics (see schemas/metrics.py).
    """
    try:
        record = loader.get_experiment(experiment_id)
        history = loader.get_history(experiment_id)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MalformedArtifactError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    series = list(history[0].keys()) if history else []
    return MetricsResponse(
        experiment_id=experiment_id,
        agent=record.agent,
        total_episodes=len(history),
        series=series,
        episodes=history,
    )


def _best_per_agent(loader: ResultsLoader) -> Dict[str, Tuple[float, ExperimentRecord]]:
    best: Dict[str, Tuple[float, ExperimentRecord]] = {}
    for record in loader.list_experiments():
        win_rate = record.summary.get("win_rate")
        if not isinstance(win_rate, (int, float)):
            continue
        current = best.get(record.agent)
        if current is None or win_rate > current[0]:
            best[record.agent] = (win_rate, record)
    return best


def _artifact_entries(loader: ResultsLoader) -> List[LeaderboardEntry]:
    entries = []
    for agent, (win_rate, record) in _best_per_agent(loader).items():
        entries.append(
            LeaderboardEntry(
                agent=agent,
                win_rate=win_rate,
                avg_episode_length=record.summary.get("avg_episode_length"),
                avg_reward=record.summary.get("avg_reward"),
                source="experiment_artifact",
                experiment_id=record.id,
            )
        )
    return entries


def _sort_key(entry: LeaderboardEntry) -> float:
    return entry.win_rate if entry.win_rate is not None else -1.0


@router.get("/leaderboard", response_model=List[LeaderboardEntry], tags=["leaderboard"])
def get_leaderboard(loader: ResultsLoader = Depends(get_results_loader)) -> List[LeaderboardEntry]:
    """Rank every cataloged agent by its best known win rate, highest first.

    DQN/PPO entries (`source="experiment_artifact"`) are computed live from
    `rl/results/` -- the best win rate found across all of that agent's
    discovered experiments. Random/CSP/Q-Learning entries
    (`source="static_reference"`) have no artifacts to read and use the
    project README's last-recorded figures instead; see module docstring.
    """
    entries = _artifact_entries(loader) + list(_STATIC_REFERENCE_ENTRIES)
    return sorted(entries, key=_sort_key, reverse=True)
