"""Schemas for experiment listing/detail endpoints (`GET /api/experiments[/{id}]`).

`GET /api/experiments` returns one row per *family* (2+ related runs sharing
an ablation-style id prefix, e.g. "exp") or standalone run -- not one row
per raw artifact -- so siblings like exp_A_baseline...exp_E_combined show up
grouped instead of as 5 individually indistinguishable entries. See
`services/results_loader.py`'s `group_experiments` for how grouping works.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RunBrief(BaseModel):
    """One run's brief info within a family's (or standalone entry's own) `runs` list."""

    id: str = Field(..., description="The individual run's own id, e.g. \"exp_A_baseline\" -- fetch full detail via GET /api/experiments/{id}.")
    title: str = Field(
        ...,
        description="Mechanically derived. Uses the run's variant suffix when it's part of a family "
        "(e.g. \"DQN - LR Decay\") so siblings don't share an identical title -- see derive_run_title.",
    )
    variant: Optional[str] = Field(None, description="The id's variant suffix (e.g. \"lr_decay\"), null for standalone runs.")
    episodes: int
    timestamp: str = Field(..., description="ISO 8601. Filesystem mtime of the artifact, not an authored field.")
    win_rate: Optional[float] = None
    avg_reward: Optional[float] = None
    metrics_available: bool


class MetricsSummary(BaseModel):
    """Aggregated evaluation metrics across a family's (or standalone entry's) runs.

    Computed only from runs that actually recorded a `win_rate` -- never
    invented for runs with no summary. All `None`/zero when no run in the
    group has one.
    """

    best_run_id: Optional[str] = None
    best_win_rate: Optional[float] = None
    avg_win_rate: Optional[float] = None
    runs_with_metrics: int = 0


class ExperimentSummary(BaseModel):
    """One row of `GET /api/experiments` -- a family (`run_count > 1`) or a standalone run (`run_count == 1`)."""

    id: str = Field(..., description="Family id (e.g. \"exp\") for a grouped family, or the run's own id when standalone.")
    title: str = Field(
        ...,
        description="Mechanically derived -- no authored experiment/family name exists in the artifacts. "
        "For a family: \"{agent} - {run_count} runs\". For standalone: the run's own title.",
    )
    agent: str = Field(..., description="\"DQN\" or \"PPO\", inferred from the artifact filename prefix.")
    algorithm: str = Field(..., description="Human-readable algorithm name, e.g. \"Double DQN\".")
    description: str = Field(..., description="Mechanically composed from real fields -- not a written summary.")
    techniques: List[str] = Field(
        default_factory=list,
        description="Union of techniques across every run in the group, derived from real recorded config flags.",
    )
    board: str = Field(..., description="e.g. \"5x5\" -- a project-wide constant, not read per-experiment (see README).")
    mines: int
    episodes_range: List[int] = Field(
        ..., description="[min, max] episode count across the group's runs -- both equal when every run trained the same length."
    )
    run_count: int
    metrics_summary: MetricsSummary
    runs: List[RunBrief] = Field(..., description="Every run in this family (or, for a standalone entry, itself alone).")


class EvaluationMetrics(BaseModel):
    """Final-evaluation numbers from an experiment's summary, if one exists."""

    win_rate: Optional[float] = None
    avg_reward: Optional[float] = None
    avg_episode_length: Optional[float] = None
    failures: Optional[int] = None
    eval_episodes: Optional[int] = None


class ArtifactManifest(BaseModel):
    """Which artifact files were actually found on disk for one experiment.

    `checkpoint_dir`/`*_checkpoint_file` are only resolved for the
    `checkpoints_{episodes}` naming convention `dqn_experiment.py`/
    `ppo_experiment.py` use -- `evaluate_agents.py`'s loose output uses
    unrelated hardcoded directory names (`checkpoints_evaluate_agents`) with
    no derivable link back to the history file's stem, so those correctly
    resolve to null rather than being guessed.
    """

    history_json: bool
    history_csv: bool
    summary_json: bool
    checkpoint_dir: Optional[str] = None
    best_checkpoint_file: Optional[str] = None
    final_checkpoint_file: Optional[str] = None


class ExperimentDetail(BaseModel):
    """Full detail for `GET /api/experiments/{id}` when `{id}` is an individual run
    (not a family id -- a family id instead returns `ExperimentSummary`, which already
    carries a full `runs` list)."""

    id: str
    agent: str
    episodes: int
    board: str
    mines: int
    seed: Optional[int] = Field(
        None, description="Only populated when actually present in the artifact; never guessed from a script default."
    )
    timestamp: str
    status: str = Field("completed", description="Always \"completed\" -- these scripts only write output after a full run.")
    has_summary: bool
    algorithm: str
    title: str = Field(..., description="Uses this run's variant suffix when it belongs to a family -- see RunBrief.title.")
    metrics_available: bool
    techniques: List[str] = Field(default_factory=list)
    family_id: Optional[str] = Field(
        None, description="The family this run belongs to (GET /api/experiments/{family_id}), or null if it's standalone."
    )
    architecture: str = Field(..., description="Short description of the network architecture used.")
    description: str = Field(
        ..., description="Mechanically composed from real fields (algorithm, episodes, board) -- not a written summary."
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Learning hyperparameters (lr, gamma, clip_epsilon, network_size, ...)."
    )
    training_configuration: Dict[str, Any] = Field(
        default_factory=dict, description="Run configuration (episodes, checkpoint_every, used_checkpoint, reward_mode, ...)."
    )
    evaluation_metrics: EvaluationMetrics = Field(default_factory=EvaluationMetrics)
    best_checkpoint: Optional[Dict[str, Any]] = Field(
        None, description="best_checkpoint_metadata from the summary, if the run recorded one."
    )
    artifacts: ArtifactManifest


class AblationMember(BaseModel):
    """One sibling experiment within an ablation family (`GET /api/experiments/{id}/ablation`)."""

    id: str
    variant_label: str = Field(..., description="The single-letter variant tag parsed from the id, e.g. \"A\", \"E\".")
    variant: str = Field(..., description="The descriptive suffix parsed from the id, e.g. \"baseline\", \"combined\".")
    title: str
    win_rate: Optional[float] = None
    avg_reward: Optional[float] = None
    avg_episode_length: Optional[float] = None


class AblationResponse(BaseModel):
    """Response for `GET /api/experiments/{id}/ablation`."""

    group: Optional[str] = Field(
        None, description="Shared id prefix this experiment was grouped under, or null if it doesn't match the pattern."
    )
    members: List[AblationMember] = Field(
        default_factory=list, description="Every experiment discovered in the same group, including this one, id-sorted."
    )
