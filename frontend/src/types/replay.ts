/** Mirrors `backend/app/schemas/replay.py`. */

export interface ReplayAction {
  row: number;
  col: number;
}

/**
 * One recorded timestep. `board_state` is the observation *after* this
 * action -- exactly what the agent would see next, nothing further ahead.
 * `reasoning` is agent-specific and shaped differently per agent (DQN:
 * `{ q_value }`, PPO: `{ action_probability }`, CSP: `{ deduction_type,
 * constraint_cells, remaining_mines, mine_probability, inference }`) --
 * `null` when nothing was recorded for that agent type (e.g. Random). Never
 * fabricate a display when this is null; show the "not recorded" fallback.
 */
export interface ReplayStep {
  step: number;
  board_state: number[][];
  action: ReplayAction;
  reward: number;
  done: boolean;
  reasoning: Record<string, unknown> | null;
}

export interface ReplaySummary {
  id: string;
  agent: string;
  experiment_id: string | null;
  won: boolean;
  /** Total number of steps the episode took. */
  steps: number;
}

export interface ReplayDetail {
  id: string;
  agent: string;
  experiment_id: string | null;
  board_size: string;
  mines: number;
  seed: number | null;
  episode_number: number | null;
  generated_at: string | null;
  /** The all-hidden board before any action -- shown at "step 0". */
  initial_board: number[][];
  timeline: ReplayStep[];
  won: boolean;
  total_reward: number;
  steps: number;
}
