"""`GET /api/experiments` and `GET /api/experiments/{id}`.

`GET /api/experiments` lists *families* (see `services/results_loader.py`'s
`group_experiments`) -- one row per group of related runs, or per standalone
run when an artifact has no siblings. `GET /api/experiments/{id}` resolves
`{id}` as a family id first (returning the same grouped shape, with its full
`runs` list) and only falls back to treating it as an individual run id
when no family matches -- so every run id that worked before this refactor
still works exactly the same way, just with an added `family_id` field.

The per-experiment training-history endpoint (`/{id}/metrics`) lives in
`routes/metrics.py` instead, alongside the leaderboard -- both are
metrics-shaped aggregates over the same underlying data this module exposes
as summaries/detail.
"""

from __future__ import annotations

from typing import List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.experiment import (
    AblationMember,
    AblationResponse,
    ArtifactManifest,
    EvaluationMetrics,
    ExperimentDetail,
    ExperimentSummary,
    MetricsSummary,
    RunBrief,
)
from app.services.results_loader import (
    ExperimentGroup,
    ExperimentNotFoundError,
    ExperimentRecord,
    MalformedArtifactError,
    ResultsLoader,
    algorithm_info,
    build_artifact_manifest,
    derive_description,
    derive_run_title,
    derive_techniques,
    get_results_loader,
    humanize_variant,
    parse_ablation_id,
    split_summary_fields,
)

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


def _group_agent(group: ExperimentGroup) -> str:
    """The agent shared by every run in the group, or a "/"-joined label on the
    (currently never observed) case where a group mixes agents."""
    agents = {run.agent for run in group.runs}
    if len(agents) == 1:
        return next(iter(agents))
    return "/".join(sorted(agents))


def _group_algorithm(group: ExperimentGroup) -> str:
    algorithms = {algorithm_info(run.agent)["algorithm"] for run in group.runs}
    if len(algorithms) == 1:
        return next(iter(algorithms))
    return " / ".join(sorted(algorithms))


def _run_variant(record: ExperimentRecord, group: ExperimentGroup) -> Optional[str]:
    # Only meaningful within an actual multi-run family -- a standalone
    # entry's own id may still happen to match the ablation pattern (a
    # lone member of what would be a family, folded back to standalone by
    # `group_experiments`), but there's no sibling to distinguish it from.
    if len(group.runs) <= 1:
        return None
    parsed = parse_ablation_id(record.id)
    return parsed["variant"] if parsed else None


def to_run_brief(record: ExperimentRecord, group: ExperimentGroup) -> RunBrief:
    variant = _run_variant(record, group)
    win_rate = record.summary.get("win_rate")
    avg_reward = record.summary.get("avg_reward")
    return RunBrief(
        id=record.id,
        title=derive_run_title(record.agent, record.episodes, variant),
        variant=variant,
        episodes=record.episodes,
        timestamp=record.timestamp,
        win_rate=win_rate if isinstance(win_rate, (int, float)) else None,
        avg_reward=avg_reward if isinstance(avg_reward, (int, float)) else None,
        metrics_available=record.metrics_available,
    )


def _metrics_summary(runs: List[ExperimentRecord]) -> MetricsSummary:
    win_rates = [(run.id, run.summary.get("win_rate")) for run in runs if isinstance(run.summary.get("win_rate"), (int, float))]
    if not win_rates:
        return MetricsSummary()
    best_run_id, best_win_rate = max(win_rates, key=lambda pair: pair[1])
    avg_win_rate = sum(win_rate for _, win_rate in win_rates) / len(win_rates)
    return MetricsSummary(
        best_run_id=best_run_id,
        best_win_rate=best_win_rate,
        avg_win_rate=avg_win_rate,
        runs_with_metrics=len(win_rates),
    )


def to_group_summary(group: ExperimentGroup) -> ExperimentSummary:
    runs = group.runs
    agent = _group_agent(group)
    algorithm = _group_algorithm(group)
    board = runs[0].board
    mines = runs[0].mines
    episodes_range = [min(run.episodes for run in runs), max(run.episodes for run in runs)]
    techniques = list(dict.fromkeys(technique for run in runs for technique in derive_techniques(run.agent, run.summary)))
    run_briefs = [to_run_brief(run, group) for run in runs]

    if len(runs) == 1:
        title = runs[0].title
        description = derive_description(algorithm, runs[0].episodes, board, mines)
    else:
        title = f"{agent} - {len(runs)} runs"
        variant_titles = [humanize_variant(v) for run in runs if (v := _run_variant(run, group))]
        if variant_titles:
            description = f"{len(runs)} {agent} runs on a {board} board ({mines} mines), comparing: {', '.join(variant_titles)}."
        else:
            description = f"{len(runs)} {agent} runs on a {board} board ({mines} mines)."

    return ExperimentSummary(
        id=group.id,
        title=title,
        agent=agent,
        algorithm=algorithm,
        description=description,
        techniques=techniques,
        board=board,
        mines=mines,
        episodes_range=episodes_range,
        run_count=len(runs),
        metrics_summary=_metrics_summary(runs),
        runs=run_briefs,
    )


def to_detail(record: ExperimentRecord, loader: ResultsLoader) -> ExperimentDetail:
    fields = split_summary_fields(record.summary)
    info = algorithm_info(record.agent)

    family_id: Optional[str] = None
    parsed = parse_ablation_id(record.id)
    if parsed is not None:
        family = loader.find_family(parsed["group"])
        if family is not None:
            family_id = family.id

    return ExperimentDetail(
        id=record.id,
        agent=record.agent,
        episodes=record.episodes,
        board=record.board,
        mines=record.mines,
        seed=record.seed,
        timestamp=record.timestamp,
        status=record.status,
        has_summary=record.has_summary,
        title=record.title if family_id is None else derive_run_title(record.agent, record.episodes, parsed["variant"] if parsed else None),
        metrics_available=record.metrics_available,
        techniques=record.techniques,
        family_id=family_id,
        algorithm=info["algorithm"],
        architecture=info["architecture"],
        description=derive_description(info["algorithm"], record.episodes, record.board, record.mines),
        hyperparameters=fields["hyperparameters"],
        training_configuration=fields["training_configuration"],
        evaluation_metrics=EvaluationMetrics(**fields["evaluation_metrics"]),
        best_checkpoint=fields["best_checkpoint"],
        artifacts=ArtifactManifest(**build_artifact_manifest(record)),
    )


@router.get("", response_model=List[ExperimentSummary])
def list_experiments(loader: ResultsLoader = Depends(get_results_loader)) -> List[ExperimentSummary]:
    """List every experiment family (or standalone run) discoverable under rl/results/.
    Never errors -- an empty or missing results directory simply yields an empty list."""
    return [to_group_summary(group) for group in loader.list_grouped_experiments()]


@router.get("/{experiment_id}", response_model=Union[ExperimentSummary, ExperimentDetail])
def get_experiment(experiment_id: str, loader: ResultsLoader = Depends(get_results_loader)) -> Union[ExperimentSummary, ExperimentDetail]:
    """`{experiment_id}` is tried as a family id first (2+ related runs, e.g. "exp"),
    returning the grouped summary shape with its full `runs` list. If no family
    matches, it's treated as an individual run id and full run detail is returned
    instead -- algorithm, architecture, hyperparameters, training configuration,
    and evaluation metrics."""
    family = loader.find_family(experiment_id)
    if family is not None:
        return to_group_summary(family)

    try:
        record = loader.get_experiment(experiment_id)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MalformedArtifactError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return to_detail(record, loader)


@router.get("/{experiment_id}/ablation", response_model=AblationResponse)
def get_experiment_ablation(experiment_id: str, loader: ResultsLoader = Depends(get_results_loader)) -> AblationResponse:
    """Sibling experiments sharing `experiment_id`'s ablation group, for comparison.

    Grouping is a generic filename pattern (`<prefix>_<LETTER>_<variant>`,
    see `parse_ablation_id`) -- not specific to any one experiment family,
    so it clusters both the DQN (`exp_A_baseline`...`exp_E_combined`) and
    PPO (`ppo_exp_A_baseline`...`ppo_exp_D_shaped_checkpoint`) ablation runs
    the same way. Returns an empty `members` list (not an error) when the id
    doesn't match the pattern, or has no siblings -- this is a normal "not
    part of an ablation family" outcome, not a failure.
    """
    try:
        record = loader.get_experiment(experiment_id)
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MalformedArtifactError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    parsed = parse_ablation_id(record.id)
    if parsed is None:
        return AblationResponse(group=None, members=[])

    members = [
        AblationMember(
            id=sibling.id,
            variant_label=(parse_ablation_id(sibling.id) or {}).get("variant_label", ""),
            variant=(parse_ablation_id(sibling.id) or {}).get("variant", ""),
            title=sibling.title,
            win_rate=sibling.summary.get("win_rate"),
            avg_reward=sibling.summary.get("avg_reward"),
            avg_episode_length=sibling.summary.get("avg_episode_length"),
        )
        for sibling in loader.list_ablation_group(parsed["group"])
    ]
    return AblationResponse(group=parsed["group"], members=members)
