/** Mirrors `backend/app/schemas/race.py`.
 *
 * A "race" is several agents' full timelines recorded on the *same* seed
 * (see `rl/evaluation/generate_race.py`) -- so every agent in one race
 * shares an identical mine layout. Reuses `ReplayStep` from `types/replay.ts`
 * for each agent's per-step timeline, same as the backend reuses
 * `schemas/replay.py`'s `ReplayStep`.
 */

import type { ReplayStep } from "@/types/replay";

export interface RaceAgentResult {
  experiment_id: string | null;
  steps: ReplayStep[];
  won: boolean;
  total_reward: number;
  /** Total number of steps this agent's episode took. */
  steps_taken: number;
}

export interface RaceSummary {
  id: string;
  seed: number;
  board_size: string;
  mines: number;
  /** Display names of the agents recorded in this race, e.g. ["Random", "CSP", "DQN", "PPO"]. */
  agents: string[];
  generated_at: string | null;
}

export interface RaceDetail {
  id: string;
  seed: number;
  board_size: string;
  mines: number;
  generated_at: string | null;
  /** The all-hidden board before any agent acted -- identical for every agent in this race. */
  initial_board: number[][];
  agents: Record<string, RaceAgentResult>;
}
