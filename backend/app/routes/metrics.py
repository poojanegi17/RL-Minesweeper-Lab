"""`GET /api/experiments/{id}/metrics` and `GET /api/leaderboard`.

Both are aggregate/derived views over experiment data rather than the raw
catalog or detail records `routes/experiments.py` and `routes/agents.py`
expose, hence living together here.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query

from app.board_levels import DEFAULT_DENSITY, DEFAULT_LEVEL, is_valid, resolve_level_dir
from app.schemas.metrics import LeaderboardEntry, MetricsResponse
from app.services.board_result_loader import BoardResultLoader
from app.services.results_loader import (
    ExperimentNotFoundError,
    ExperimentRecord,
    MalformedArtifactError,
    ResultsLoader,
    get_results_loader,
)

router = APIRouter(prefix="/api", tags=["metrics"])

# Mirrors routes/agents.py's catalog -- kept as its own small constant here
# rather than imported, to avoid a cross-route import for one name list.
_AGENT_NAMES = ["Random", "CSP", "Q-Learning", "DQN", "PPO"]

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


def _board_result_entries(board_loader: BoardResultLoader) -> List[LeaderboardEntry]:
    """Leaderboard rows for a non-default (level, density) -- one row per
    agent, sourced from `evaluate_board_config.py`'s output. An agent with no
    board-result file at this level (DQN/PPO before that level is trained)
    still gets a row, tagged `source="not_trained"` with `win_rate=None`,
    rather than silently disappearing from the list."""
    results_by_agent = {r["agent"]: r for r in board_loader.list_results()}
    entries = []
    for agent in _AGENT_NAMES:
        result = results_by_agent.get(agent)
        if result is None:
            entries.append(LeaderboardEntry(agent=agent, win_rate=None, avg_episode_length=None, avg_reward=None, source="not_trained"))
        else:
            entries.append(
                LeaderboardEntry(
                    agent=agent,
                    win_rate=result["win_rate"],
                    avg_episode_length=result["avg_episode_length"],
                    avg_reward=result["avg_reward"],
                    source="board_result",
                )
            )
    return entries


@router.get("/leaderboard", response_model=List[LeaderboardEntry], tags=["leaderboard"])
def get_leaderboard(
    level: str = Query(DEFAULT_LEVEL, description="Difficulty level, e.g. \"beginner\"."),
    density: str = Query(DEFAULT_DENSITY, description="Mine-density preset, e.g. \"standard\"."),
    loader: ResultsLoader = Depends(get_results_loader),
) -> List[LeaderboardEntry]:
    """Rank every cataloged agent by its best known win rate, highest first,
    at the given `(level, density)` (see `GET /api/board-configs`).

    At the default "beginner"/"standard" board (every experiment/replay/race
    this project has generated to date), DQN/PPO entries
    (`source="experiment_artifact"`) are computed live from `rl/results/` --
    the best win rate found across all of that agent's discovered
    experiments. Random/CSP/Q-Learning entries (`source="static_reference"`)
    have no artifacts to read and use the project README's last-recorded
    figures instead; see module docstring.

    At any other level/density, every entry (`source="board_result"`, or
    `"not_trained"` if that agent has no data there yet) comes from
    `evaluate_board_config.py`'s output instead -- a different, much simpler
    artifact shape than a training experiment, since most of these runs
    involve no training at all (see `services/board_result_loader.py`).
    """
    if not is_valid(level, density):
        raise HTTPException(status_code=400, detail=f"Unknown level/density: {level!r}/{density!r}")

    if level == DEFAULT_LEVEL and density == DEFAULT_DENSITY:
        entries = _artifact_entries(loader) + list(_STATIC_REFERENCE_ENTRIES)
    else:
        scoped_dir = resolve_level_dir(loader.results_dir, level, density)
        entries = _board_result_entries(BoardResultLoader(scoped_dir))

    return sorted(entries, key=_sort_key, reverse=True)
