/** Mirrors `backend/app/schemas/metrics.py`. */

/**
 * One row of an experiment's training history. DQN and PPO log different
 * diagnostic fields (see `MetricsResponse.series` for which ones are
 * actually present on a given experiment) -- only the fields common to
 * both are typed explicitly; everything else is indexed loosely.
 */
export interface MetricPoint {
  episode: number;
  total_reward: number;
  steps: number;
  won: boolean;
  [metric: string]: number | boolean | string | null | undefined;
}

export interface MetricsResponse {
  experiment_id: string;
  agent: string;
  total_episodes: number;
  series: string[];
  episodes: MetricPoint[];
}

export interface LeaderboardEntry {
  agent: string;
  win_rate: number | null;
  avg_episode_length: number | null;
  avg_reward: number | null;
  /** At the default beginner/standard board: "experiment_artifact" (read
   * live from rl/results/) or "static_reference" (README-recorded, no
   * artifact exists). At any other level/density: "board_result" (read from
   * evaluate_board_config.py's output) or "not_trained" (no data yet). */
  source: "experiment_artifact" | "static_reference" | "board_result" | "not_trained";
  experiment_id: string | null;
}
