/** Mirrors `backend/app/schemas/experiment.py`.
 *
 * `GET /api/experiments` groups related runs into families (2+ runs sharing
 * an ablation-style id prefix, e.g. "exp_A_baseline"..."exp_E_combined" ->
 * family "exp") or presents a standalone run the same shape with
 * `run_count: 1` -- see `ExperimentSummary`. `GET /api/experiments/{id}`
 * returns that same grouped shape when `{id}` is a family id, or full
 * `ExperimentDetail` for an individual run id -- see `isFamilySummary` in
 * `@/lib/experimentAdapters` for how the frontend tells the two apart.
 */

/** One run's brief info within a family's (or standalone entry's own) `runs` list. */
export interface RunBrief {
  /** The individual run's own id, e.g. "exp_A_baseline" -- fetch full detail via GET /api/experiments/{id}. */
  id: string;
  /** Mechanically derived. Uses the run's variant suffix when it's part of a
   * family (e.g. "DQN - LR Decay") so siblings don't share an identical title. */
  title: string;
  /** The id's variant suffix (e.g. "lr_decay"), null for standalone runs. */
  variant: string | null;
  episodes: number;
  /** ISO 8601. Filesystem mtime of the artifact, not an authored field. */
  timestamp: string;
  win_rate: number | null;
  avg_reward: number | null;
  metrics_available: boolean;
  /** Which board distribution this run trained under: "v1" (mines placed
   * before the first click, so the opening move can lose) or "v2"
   * (`first_click_safe: "area"`). The two are different games and their win
   * rates are not comparable, so anything picking a "best" run must compare
   * only within one -- see `bestRunForEnv`. Runs predating the flag are "v1"
   * by construction. */
  env_version: EnvVersion;
}

export type EnvVersion = "v1" | "v2";

/** Aggregated evaluation metrics across a family's (or standalone entry's) runs.
 * Computed only from runs that actually recorded a `win_rate` -- `null`/`0` when none did. */
export interface MetricsSummary {
  best_run_id: string | null;
  best_win_rate: number | null;
  avg_win_rate: number | null;
  runs_with_metrics: number;
}

/** One row of `GET /api/experiments` -- a family (`run_count > 1`) or a standalone run (`run_count === 1`). */
export interface ExperimentSummary {
  /** Family id (e.g. "exp") for a grouped family, or the run's own id when standalone. */
  id: string;
  /** Mechanically derived -- no authored experiment/family name exists in the
   * artifacts. For a family: "{agent} - {run_count} runs". For standalone: the run's own title. */
  title: string;
  /** "DQN" or "PPO", inferred from the artifact filename prefix. */
  agent: string;
  algorithm: string;
  /** Mechanically composed from real fields -- not a written summary. */
  description: string;
  /** Union of techniques across every run in the group, derived from real recorded config flags. */
  techniques: string[];
  /** e.g. "5x5" -- a project-wide constant, not read per-experiment. */
  board: string;
  mines: number;
  /** [min, max] episode count across the group's runs -- both equal when every run trained the same length. */
  episodes_range: [number, number];
  run_count: number;
  metrics_summary: MetricsSummary;
  /** Every run in this family (or, for a standalone entry, itself alone). Never empty. */
  runs: RunBrief[];
}

export interface EvaluationMetrics {
  win_rate: number | null;
  avg_reward: number | null;
  avg_episode_length: number | null;
  failures: number | null;
  eval_episodes: number | null;
  /** `[low, high]` as percentages -- a Wilson score interval. Non-null only for
   * runs re-scored by `evaluation/reevaluate_checkpoints.py`; a null here means
   * `win_rate` is still the 200-episode figure the training script wrote, which
   * at a ~1-3% win rate rests on a handful of wins. */
  win_rate_ci95: [number, number] | null;
  /** Which tool produced `win_rate`, when it wasn't the training script itself. */
  evaluation_source: string | null;
}

/** Arbitrary hyperparameter/training-config value: whatever a given experiment
 * script happened to log (numbers, strings, nested schedules, null). */
export type SummaryValue = string | number | boolean | null | SummaryValue[];

export interface BestCheckpoint {
  episode: number;
  win_rate: number;
  average_reward: number;
  average_episode_length?: number;
  timestamp: string;
}

/** Which artifact files were actually found on disk for one experiment.
 * `checkpoint_dir`/`*_checkpoint_file` are null whenever they can't be
 * derived from the `checkpoints_{episodes}` naming convention (e.g. loose
 * `evaluate_agents.py` output) -- see backend `results_loader.py`. */
export interface ArtifactManifest {
  history_json: boolean;
  history_csv: boolean;
  summary_json: boolean;
  checkpoint_dir: string | null;
  best_checkpoint_file: string | null;
  final_checkpoint_file: string | null;
}

/** Full detail for `GET /api/experiments/{id}` when `{id}` resolves to an
 * individual run (not a family id -- a family id instead returns
 * `ExperimentSummary`, which already carries a full `runs` list). Does
 * *not* extend `ExperimentSummary`: the two shapes diverged once grouping
 * was introduced (one run's own episode count vs. a family's range, etc.). */
export interface ExperimentDetail {
  id: string;
  agent: string;
  episodes: number;
  board: string;
  mines: number;
  /** Only populated when actually present in the artifact; never guessed from a script default. */
  seed: number | null;
  timestamp: string;
  status: string;
  has_summary: boolean;
  algorithm: string;
  /** Uses this run's variant suffix when it belongs to a family -- see `RunBrief.title`. */
  title: string;
  metrics_available: boolean;
  /** Board distribution this run trained under -- see `RunBrief.env_version`. */
  env_version: EnvVersion;
  techniques: string[];
  /** The family this run belongs to (fetch via `GET /api/experiments/{family_id}`), or null if standalone. */
  family_id: string | null;
  architecture: string;
  /** Mechanically composed from real fields (algorithm, episodes, board) -- not a written summary. */
  description: string;
  hyperparameters: Record<string, SummaryValue>;
  training_configuration: Record<string, SummaryValue>;
  evaluation_metrics: EvaluationMetrics;
  best_checkpoint: BestCheckpoint | null;
  artifacts: ArtifactManifest;
}

/** One sibling experiment within an ablation family (`GET /api/experiments/{id}/ablation`).
 * Superseded for UI purposes by `ExperimentSummary.runs`/`RunBrief` (the same grouping,
 * available directly on a family without a second request) -- kept because the backend
 * endpoint still exists and returns this shape. */
export interface AblationMember {
  id: string;
  variant_label: string;
  variant: string;
  title: string;
  win_rate: number | null;
  avg_reward: number | null;
  avg_episode_length: number | null;
}

export interface AblationResponse {
  /** Null when the experiment's id doesn't match the ablation-family naming pattern. */
  group: string | null;
  /** Every experiment discovered in the same group, including the queried one, id-sorted.
   * Empty when `group` is null, or when the experiment has no siblings. */
  members: AblationMember[];
}
