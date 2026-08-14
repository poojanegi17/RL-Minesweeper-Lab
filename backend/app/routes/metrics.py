"""`GET /api/experiments/{id}/metrics` and `GET /api/leaderboard`.

Both are aggregate/derived views over experiment data rather than the raw
catalog or detail records `routes/experiments.py` and `routes/agents.py`
expose, hence living together here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import BENCHMARK_BOARD_COLS, BENCHMARK_BOARD_MINES, BENCHMARK_BOARD_ROWS
from app.board_levels import (
    DEFAULT_DENSITY,
    DEFAULT_LEVEL,
    is_valid,
    is_valid_first_click_policy,
    resolve_level_dir,
    resolve_policy_level_dir,
)
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
# services/results_loader.py), so there is nothing under rl/results/ for these
# three to be measured from live at the default board. They're included here so
# the leaderboard covers the full agent catalog, and tagged
# `source="static_reference"` below rather than presented as if freshly read
# from a file, since they aren't.
#
# All three are 2,000-episode figures at seed 42, hardcoded but reproducible by
# re-running the script named per row rather than hand-transcribed.
#
# All three are **v1**: measured with `first_click_safe="none"`, so the opening
# click can itself be a mine. That matches the distribution the default board's
# DQN/PPO entries are measured under (`_measures_default_board` filters the
# `experiment_artifact` path to 5x5/v1), so every row on the default leaderboard
# now describes the same game and the ranking is like-for-like.
#
# These previously held the v2 figures (Q-Learning 71.70%, CSP 70.35%, Random
# 1.30%) while DQN/PPO were read from v1 training runs -- a mixed axis that
# ranked a v2 agent above a v1 one as though the gap had been measured. The v2
# numbers are not lost: they are the committed board results under
# `results_public/v2/levels/beginner/standard/`, which is what the UI serves
# whenever a first-click policy is explicitly selected.
#
# Values mirror `results_public/v1/levels/beginner/standard/*_board_result.json`
# exactly. Regenerate with:
#     python -m evaluation.rebaseline_board_configs --first-click-safe none
#
# Q-Learning's 100,000-episode training budget is recorded per result file as
# `train_episodes`; it was chosen by measurement rather than convention -- see
# the training-budget sweep in the README. Regenerate with:
#     python -m evaluation.rebaseline_board_configs --agents q_learning \
#         --levels beginner --first-click-safe none --q-train-episodes 100000
_STATIC_REFERENCE_ENTRIES: List[LeaderboardEntry] = [
    # v1: 868/2000, 95% CI [41.24, 45.58]
    LeaderboardEntry(agent="CSP", win_rate=0.434, avg_episode_length=6.6715, avg_reward=4.3515, source="static_reference"),
    # v1: 38/2000, 95% CI [1.39, 2.60], 100,000 training episodes
    LeaderboardEntry(agent="Q-Learning", win_rate=0.019, avg_episode_length=4.1725, avg_reward=-6.4475, source="static_reference"),
    # v1: 9/2000, 95% CI [0.24, 0.85]
    LeaderboardEntry(agent="Random", win_rate=0.0045, avg_episode_length=3.546, avg_reward=-7.364, source="static_reference"),
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
    # `record.episodes` (the run's own summary) rather than `len(history)`:
    # committed histories are subsampled for size (see
    # `rl/evaluation/compact_public_histories.py`), so the number of rows in
    # the file is the chart's resolution, not the length of the run. Each row
    # keeps its true `episode` number, so the x-axis stays correct either way.
    # Falls back to the row count for loose artifacts that carry no summary.
    return MetricsResponse(
        experiment_id=experiment_id,
        agent=record.agent,
        total_episodes=record.episodes or len(history),
        series=series,
        episodes=history,
    )


def _measures_default_board(record: ExperimentRecord) -> bool:
    """Whether this run's win rate describes the *default* leaderboard board:
    the 5x5 benchmark with 5 mines, under the v1 distribution.

    Without this filter `_best_per_agent` ranks every discovered run against
    every other regardless of what board it was measured on, which was harmless
    only while every run in the project was 5x5/v1. It no longer is:
    `dqn_intermediate_E_env_v2` scores 80.15% on a *9x9* board under v2, beats
    every genuine 5x5 result, and was therefore being served as DQN's
    beginner/standard figure -- a number from a different board size and a
    different game presented as this one's headline.

    Runs predating the `--first-click-safe` flag record no `env_version` and
    are v1 by construction, matching `routes/experiments.run_env_version`.
    """
    if record.board != f"{BENCHMARK_BOARD_ROWS}x{BENCHMARK_BOARD_COLS}" or record.mines != BENCHMARK_BOARD_MINES:
        return False
    return record.summary.get("env_version", "v1") == "v1"


def _best_per_agent(loader: ResultsLoader) -> Dict[str, Tuple[float, ExperimentRecord]]:
    best: Dict[str, Tuple[float, ExperimentRecord]] = {}
    for record in loader.list_experiments():
        win_rate = record.summary.get("win_rate")
        if not isinstance(win_rate, (int, float)) or not _measures_default_board(record):
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
    first_click_safe: Optional[str] = Query(
        None,
        description=(
            'Read from a specific board distribution: "area" (opening click opens a mine-free '
            '3x3 block) or "none" (mines placed before the first click, so it can lose). '
            "Omit for the default tree. The two are different games and their win rates are "
            "not comparable."
        ),
    ),
    loader: ResultsLoader = Depends(get_results_loader),
) -> List[LeaderboardEntry]:
    """Rank every cataloged agent by its best known win rate, highest first,
    at the given `(level, density)` (see `GET /api/board-configs`).

    At the default "beginner"/"standard" board (every experiment/replay/race
    this project has generated to date), DQN/PPO entries
    (`source="experiment_artifact"`) are computed live from `rl/results/` --
    the best win rate found across all of that agent's discovered
    experiments. Random/CSP/Q-Learning entries (`source="static_reference"`)
    have no artifacts to read and use `rescore_board_configs.py`'s
    2,000-episode figures instead; see module comment.

    At any other level/density, every entry (`source="board_result"`, or
    `"not_trained"` if that agent has no data there yet) comes from
    `evaluate_board_config.py`'s output instead -- a different, much simpler
    artifact shape than a training experiment, since most of these runs
    involve no training at all (see `services/board_result_loader.py`).
    """
    if not is_valid(level, density):
        raise HTTPException(status_code=400, detail=f"Unknown level/density: {level!r}/{density!r}")

    if first_click_safe is not None:
        # An explicit policy always reads that policy's own tree, including at
        # beginner/standard -- the artifact/static-reference path below can't
        # honour it, since those figures come from whichever distribution the
        # root directory was last re-baselined to.
        if not is_valid_first_click_policy(first_click_safe):
            raise HTTPException(status_code=400, detail=f"Unknown first_click_safe policy: {first_click_safe!r}")
        scoped_dir = resolve_policy_level_dir(loader.results_dir, first_click_safe, level, density)
        return sorted(_board_result_entries(BoardResultLoader(scoped_dir)), key=_sort_key, reverse=True)

    if level == DEFAULT_LEVEL and density == DEFAULT_DENSITY:
        entries = _artifact_entries(loader) + list(_STATIC_REFERENCE_ENTRIES)
    else:
        scoped_dir = resolve_level_dir(loader.results_dir, level, density)
        entries = _board_result_entries(BoardResultLoader(scoped_dir))

    return sorted(entries, key=_sort_key, reverse=True)
