"""Reads and validates RL experiment artifacts under `rl/results/`.

No API route touches the filesystem directly -- this module is the only
place that does, so route handlers stay thin translators from
`ExperimentRecord` to Pydantic response models.

Two kinds of artifact exist on disk today (see backend/README.md for the
full inspection this was based on):

    - Ablation-style experiment directories, one per named run
      (`exp_A_baseline/`, `ppo_exp_C_shaped/`, ...), each holding exactly
      one `{agent}_history_{episodes}.json` + `.csv` + `_summary.json`
      triplet plus a `checkpoints_{episodes}/` directory of `.pt` files.
    - Loose top-level files, some with a `_summary.json` sibling
      (`dqn_history_25000.json`), some without
      (`dqn_evaluate_agents_history.json` -- `evaluate_agents.py` never
      writes a summary).

Only Random, CSP, and Q-Learning produce *no* artifacts at all (they're
never run through `dqn_experiment.py`/`ppo_experiment.py`-style scripts, so
there's nothing under `rl/results/` to discover for them) -- see
`schemas/agent.py`'s `has_experiment_artifacts` field.

Schema drift: older summary files (from before checkpointing/network-size
options existed) are missing several keys later summaries have. Every field
beyond `episodes` is treated as optional throughout this module for exactly
that reason -- see `_split_summary_fields`.

CSV files are never read: they're a byte-for-byte export of the same
per-episode history (`training/history_export.py` writes both from the same
in-memory list), so the JSON is a strictly equivalent, better-typed source.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import BENCHMARK_BOARD_COLS, BENCHMARK_BOARD_MINES, BENCHMARK_BOARD_ROWS, settings

logger = logging.getLogger(__name__)

# Summary-JSON keys bucketed into the ExperimentDetail response's
# `training_configuration` vs. `evaluation_metrics` groups; everything else
# present in a summary (e.g. lr, gamma, network_size, clip_epsilon --
# whatever a given agent happens to log) falls through to
# `hyperparameters`, so a newly-added hyperparameter in some future
# experiment script doesn't need a matching change here.
_TRAINING_CONFIG_KEYS = {"episodes", "checkpoint_every", "used_checkpoint", "reward_mode", "train_seconds"}
_EVALUATION_KEYS = {"win_rate", "avg_episode_length", "avg_reward", "failures", "eval_episodes"}
_BEST_CHECKPOINT_KEY = "best_checkpoint_metadata"

_ALGORITHM_INFO = {
    "DQN": {
        "algorithm": "Double DQN",
        "architecture": "CNN Q-network over an 11-channel board encoding, with experience replay and a target network.",
    },
    "PPO": {
        "algorithm": "PPO (clipped surrogate objective)",
        "architecture": "Shared CNN actor-critic over an 11-channel board encoding, trained with GAE.",
    },
    "Unknown": {
        "algorithm": "Unknown",
        "architecture": "Unknown",
    },
}


class ResultsLoaderError(Exception):
    """Base class for all results-loading errors, so routes can catch one type."""


class ExperimentNotFoundError(ResultsLoaderError):
    """Raised when an experiment id doesn't match any discovered artifact."""


class MalformedArtifactError(ResultsLoaderError):
    """Raised when an artifact exists on disk but isn't valid/expected JSON."""


@dataclass(frozen=True)
class ExperimentRecord:
    """A discovered experiment, with its raw summary dict kept for the detail endpoint."""

    id: str
    agent: str
    episodes: int
    board: str
    mines: int
    seed: Optional[int]
    timestamp: str
    has_summary: bool
    summary: Dict[str, Any] = field(default_factory=dict)
    history_path: Optional[Path] = None

    @property
    def status(self) -> str:
        # Every artifact on disk today is only ever written after a full
        # training run completes (see module docstring) -- there is no
        # partial-write or failure state to report from files alone.
        return "completed"

    @property
    def metrics_available(self) -> bool:
        """Whether a history file was discovered -- GET .../metrics should succeed.

        Existence-based, same convention as `has_summary`: cheap to compute
        for every row of a listing, at the cost of not catching a history
        file that exists but turns out to be corrupt (that still surfaces
        as a 422 from `get_history`, just not predicted here).
        """
        return self.history_path is not None

    @property
    def title(self) -> str:
        return derive_title(self.agent, self.episodes)

    @property
    def techniques(self) -> List[str]:
        return derive_techniques(self.agent, self.summary)


@dataclass(frozen=True)
class ExperimentGroup:
    """A family of related runs (2+ members sharing an ablation-style id prefix,
    e.g. "exp" for exp_A_baseline...exp_E_combined) or a single standalone run
    presented the same way (`id` = the run's own id, `runs` = itself alone).
    See `group_experiments`."""

    id: str
    runs: List[ExperimentRecord]  # id-sorted, always non-empty


def _agent_from_stem(stem: str) -> str:
    if stem.startswith("dqn"):
        return "DQN"
    if stem.startswith("ppo"):
        return "PPO"
    return "Unknown"


def _iso_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


class ResultsLoader:
    """Discovers and reads experiment artifacts under a `rl/results/`-shaped directory."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)

    def _discover(self) -> Dict[str, Dict[str, Optional[Path]]]:
        """Map experiment id -> {"history": Path, "summary": Optional[Path]}.

        Returns an empty dict (not an error) when `results_dir` doesn't
        exist or holds nothing recognizable -- an empty results directory
        is a valid, expected state for a project with no experiments run
        yet, not a failure.
        """
        groups: Dict[str, Dict[str, Optional[Path]]] = {}
        if not self.results_dir.exists():
            logger.warning("results_dir %s does not exist; treating as empty", self.results_dir)
            return groups

        def _scan(directory: Path, *, namespace_by_dir: bool) -> None:
            try:
                json_files = sorted(directory.glob("*.json"))
            except OSError as exc:
                logger.warning("could not list %s: %s", directory, exc)
                return

            summaries = {p.stem[: -len("_summary")]: p for p in json_files if p.stem.endswith("_summary")}
            histories = {p.stem: p for p in json_files if not p.stem.endswith("_summary")}

            for stem, hist_path in histories.items():
                exp_id = directory.name if (namespace_by_dir and len(histories) == 1) else stem
                groups[exp_id] = {"history": hist_path, "summary": summaries.get(stem)}

        _scan(self.results_dir, namespace_by_dir=False)
        for child in sorted(self.results_dir.iterdir()):
            # "checkpoints_*" dirs hold .pt files, never experiment JSON, so
            # they're naturally skipped by the *.json glob regardless -- but
            # "replays/" *does* hold JSON files (one per recorded episode,
            # written by generate_replays.py), which would otherwise be
            # misdiscovered as bogus experiments (e.g. "ppo_episode_1" with
            # episodes=0, since a replay file has no "episodes" key and its
            # top-level shape is a dict, not the row-list history.json
            # expects). Excluded explicitly for that reason, not because it
            # doesn't parse -- see services/replay_loader.py for the module
            # that actually reads this directory.
            if child.is_dir() and not child.name.startswith("checkpoints") and child.name != "replays":
                _scan(child, namespace_by_dir=True)

        return groups

    def _load_summary(self, path: Optional[Path]) -> Dict[str, Any]:
        if path is None:
            return {}
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise MalformedArtifactError(f"Could not parse summary JSON at {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise MalformedArtifactError(f"Expected a JSON object in {path}, got {type(data).__name__}")
        return data

    def _build_record(self, experiment_id: str, paths: Dict[str, Optional[Path]]) -> ExperimentRecord:
        history_path = paths.get("history")
        summary_path = paths.get("summary")
        summary = self._load_summary(summary_path)

        stem = history_path.stem if history_path is not None else experiment_id
        agent = _agent_from_stem(stem)
        timestamp_source = summary_path or history_path
        episodes = summary.get("episodes")
        if episodes is None and history_path is not None:
            # Loose evaluate_agents.py-style history files have no summary
            # to read `episodes` from -- fall back to the actual row count,
            # which is exactly what "episodes" means for these artifacts.
            try:
                history = json.loads(history_path.read_text())
                episodes = len(history) if isinstance(history, list) else None
            except (OSError, json.JSONDecodeError):
                episodes = None

        return ExperimentRecord(
            id=experiment_id,
            agent=agent,
            episodes=episodes if episodes is not None else 0,
            board=f"{BENCHMARK_BOARD_ROWS}x{BENCHMARK_BOARD_COLS}",
            mines=BENCHMARK_BOARD_MINES,
            seed=summary.get("seed"),
            timestamp=_iso_timestamp(timestamp_source) if timestamp_source else "",
            has_summary=summary_path is not None,
            summary=summary,
            history_path=history_path,
        )

    def list_experiments(self) -> List[ExperimentRecord]:
        """Return every discoverable experiment, skipping (and logging) unparseable ones.

        A single malformed summary must not take down the whole listing --
        it's excluded here and logged, distinct from `get_experiment`, which
        raises for the one experiment actually being asked about.
        """
        records: List[ExperimentRecord] = []
        for experiment_id, paths in sorted(self._discover().items()):
            try:
                records.append(self._build_record(experiment_id, paths))
            except MalformedArtifactError as exc:
                logger.warning("skipping experiment %r in listing: %s", experiment_id, exc)
        return records

    def list_ablation_group(self, group: str) -> List[ExperimentRecord]:
        """Every discovered experiment whose id parses to ablation `group`, id-sorted.

        Built on `list_experiments()`, so a malformed sibling is silently
        excluded from the group rather than breaking the whole lookup --
        same resilience contract as the plain listing.
        """
        members = [record for record in self.list_experiments() if (parse_ablation_id(record.id) or {}).get("group") == group]
        return sorted(members, key=lambda r: r.id)

    def list_grouped_experiments(self) -> List[ExperimentGroup]:
        """Every discoverable experiment, grouped into families where the id
        pattern supports it (see `group_experiments`) and left standalone
        otherwise. This is what `GET /api/experiments` presents -- one row
        per family-or-standalone-run, not one row per raw artifact, so
        siblings like exp_A_baseline...exp_E_combined show up as a single
        grouped entry instead of 5 individually indistinguishable ones.
        """
        return group_experiments(self.list_experiments())

    def find_family(self, family_id: str) -> Optional[ExperimentGroup]:
        """Return the family (2+ runs) matching `family_id`, or None.

        None covers both "no such id at all" and "that id matched a group
        prefix but only had one member, so it was folded into standalone
        instead of being exposed as a family of one" -- both are legitimately
        "not a family", so the caller (routes/experiments.py) falls back to
        treating `family_id` as an individual run id either way.
        """
        for group in self.list_grouped_experiments():
            if group.id == family_id and len(group.runs) > 1:
                return group
        return None

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        """Return one experiment's record, or raise if it doesn't exist / can't be parsed."""
        discovered = self._discover()
        if experiment_id not in discovered:
            raise ExperimentNotFoundError(f"No experiment found with id {experiment_id!r}")
        return self._build_record(experiment_id, discovered[experiment_id])

    def get_history(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Return the full per-episode training history for one experiment."""
        record = self.get_experiment(experiment_id)
        if record.history_path is None:
            raise MalformedArtifactError(f"Experiment {experiment_id!r} has no history file on disk")
        try:
            data = json.loads(record.history_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise MalformedArtifactError(f"Could not parse history JSON at {record.history_path}: {exc}") from exc
        if not isinstance(data, list):
            raise MalformedArtifactError(f"Expected a JSON array in {record.history_path}, got {type(data).__name__}")
        return data


def split_summary_fields(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Split a raw summary dict into training_configuration / evaluation_metrics / hyperparameters / best_checkpoint.

    Any key not recognized as training-config or evaluation falls through to
    `hyperparameters` by default, so a new hyperparameter added to some
    future experiment script shows up automatically without a code change
    here (see module docstring).
    """
    training_configuration: Dict[str, Any] = {}
    evaluation_metrics: Dict[str, Any] = {}
    hyperparameters: Dict[str, Any] = {}
    best_checkpoint = summary.get(_BEST_CHECKPOINT_KEY)

    for key, value in summary.items():
        if key == _BEST_CHECKPOINT_KEY:
            continue
        elif key in _TRAINING_CONFIG_KEYS:
            training_configuration[key] = value
        elif key in _EVALUATION_KEYS:
            evaluation_metrics[key] = value
        else:
            hyperparameters[key] = value

    return {
        "training_configuration": training_configuration,
        "evaluation_metrics": evaluation_metrics,
        "hyperparameters": hyperparameters,
        "best_checkpoint": best_checkpoint,
    }


def algorithm_info(agent: str) -> Dict[str, str]:
    """Static algorithm/architecture description text for a given agent name."""
    return _ALGORITHM_INFO.get(agent, _ALGORITHM_INFO["Unknown"])


def derive_title(agent: str, episodes: int) -> str:
    """Mechanically composed -- no artifact anywhere records an authored experiment name."""
    return f"{agent} - {episodes:,} episodes"


def humanize_variant(variant: str) -> str:
    """Formats a parsed id's variant suffix (e.g. "lr_decay", from
    `parse_ablation_id`) into display text ("LR Decay") -- pure text
    formatting of a substring of the real id, never invented content. "lr"
    is special-cased to the conventional acronym capitalization; every other
    word is just title-cased.
    """
    return " ".join(word.upper() if word.lower() == "lr" else word.capitalize() for word in variant.split("_"))


def derive_run_title(agent: str, episodes: int, variant: Optional[str] = None) -> str:
    """Mechanically composed. When `variant` is given (this run is a member of
    an ablation family with 2+ real siblings), it's used instead of the
    episode count -- otherwise every sibling trained for the same number of
    episodes (the common case) would render an identical, indistinguishable
    title (e.g. every DQN ablation run showing "DQN - 25,000 episodes"). A
    variant-based title ("DQN - LR Decay") is still 100% derived from the
    run's own id, not invented.
    """
    if variant:
        return f"{agent} - {humanize_variant(variant)}"
    return derive_title(agent, episodes)


def derive_description(algorithm: str, episodes: int, board: str, mines: int) -> str:
    """Mechanically composed from real fields -- not a written summary of results."""
    return f"{algorithm} trained for {episodes:,} episodes on a {board} board ({mines} mines)."


# Each (label, predicate) pair maps directly to a real, recorded config
# value -- never a claim about something not actually present in the
# summary. The first rule per agent is a structural fact about this
# codebase's implementation (DQNAgent always runs Double DQN;
# PPOAgent always uses the clipped-surrogate objective -- see
# `rl/agents/dqn_agent.py`/`ppo_agent.py`), true for every experiment
# regardless of its specific config, so it's unconditional.
_TECHNIQUE_RULES: Dict[str, List[Any]] = {
    "DQN": [
        ("Double DQN", lambda summary: True),
        ("LR decay", lambda summary: summary.get("lr_schedule") is not None),
        ("Reduced network capacity", lambda summary: summary.get("network_size") == "small"),
        ("Best-checkpoint deployment", lambda summary: str(summary.get("used_checkpoint", "")).startswith("best")),
    ],
    "PPO": [
        ("Clipped-surrogate PPO", lambda summary: True),
        ("Reward shaping", lambda summary: summary.get("reward_mode") == "shaped"),
        ("Best-checkpoint deployment", lambda summary: str(summary.get("used_checkpoint", "")).startswith("best")),
    ],
}


def derive_techniques(agent: str, summary: Dict[str, Any]) -> List[str]:
    """Real, recorded config flags rendered as short labels -- see `_TECHNIQUE_RULES`."""
    rules = _TECHNIQUE_RULES.get(agent, [])
    return [label for label, predicate in rules if predicate(summary)]


def build_artifact_manifest(record: ExperimentRecord) -> Dict[str, Any]:
    """Which artifact files actually exist on disk for `record` -- see `ArtifactManifest`'s docstring."""
    history_json = record.history_path is not None
    history_csv = bool(record.history_path and record.history_path.with_suffix(".csv").exists())

    checkpoint_dir: Optional[Path] = None
    if record.history_path is not None:
        candidate = record.history_path.parent / f"checkpoints_{record.episodes}"
        if candidate.is_dir():
            checkpoint_dir = candidate

    def _first_match(pattern: str) -> Optional[str]:
        if checkpoint_dir is None:
            return None
        matches = sorted(checkpoint_dir.glob(pattern))
        return matches[0].name if matches else None

    return {
        "history_json": history_json,
        "history_csv": history_csv,
        "summary_json": record.has_summary,
        "checkpoint_dir": checkpoint_dir.name if checkpoint_dir else None,
        "best_checkpoint_file": _first_match("best_*.pt"),
        "final_checkpoint_file": _first_match("final_*.pt"),
    }


# Matches ablation-style ids like "exp_A_baseline" (group="exp", variant_label="A",
# variant="baseline") or "ppo_exp_C_shaped" (group="ppo_exp", variant_label="C",
# variant="shaped") -- generic over any id following this "<prefix>_<LETTER>_<variant>"
# convention, not hardcoded to any specific experiment family.
_ABLATION_ID_PATTERN = re.compile(r"^(?P<group>.+?)_(?P<variant_label>[A-Z])_(?P<variant>.+)$")


def parse_ablation_id(experiment_id: str) -> Optional[Dict[str, str]]:
    """Parse an id into its ablation group/variant, or None if it doesn't match the pattern."""
    match = _ABLATION_ID_PATTERN.match(experiment_id)
    if not match:
        return None
    return match.groupdict()


def group_experiments(records: List[ExperimentRecord]) -> List[ExperimentGroup]:
    """Bucket `records` into families sharing an ablation-style id prefix (2+
    real members, via `parse_ablation_id`) or leave them standalone.

    A prefix with only one matching record is *not* turned into a family of
    one -- there's no data on disk today where that happens (every id
    matching the "<prefix>_<LETTER>_<variant>" pattern currently has 2+
    siblings), but if it ever did, presenting a single run under a
    synthetic group id would be more confusing than useful, so it's folded
    back into the standalone list under its own real id instead.

    Returned id-sorted by group id, so listing order is stable.
    """
    by_prefix: Dict[str, List[ExperimentRecord]] = {}
    standalone: List[ExperimentRecord] = []

    for record in records:
        parsed = parse_ablation_id(record.id)
        if parsed is not None:
            by_prefix.setdefault(parsed["group"], []).append(record)
        else:
            standalone.append(record)

    groups: List[ExperimentGroup] = []
    for prefix, members in by_prefix.items():
        if len(members) >= 2:
            groups.append(ExperimentGroup(id=prefix, runs=sorted(members, key=lambda r: r.id)))
        else:
            standalone.extend(members)

    groups.extend(ExperimentGroup(id=record.id, runs=[record]) for record in standalone)
    return sorted(groups, key=lambda g: g.id)


def get_results_loader() -> ResultsLoader:
    """FastAPI dependency provider. Tests override this via `app.dependency_overrides`
    to point at a temp directory instead of the real `rl/results/`."""
    return ResultsLoader(settings.results_dir)
