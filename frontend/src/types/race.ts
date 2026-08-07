/** Mirrors `backend/app/schemas/race.py`.
 *
 * A "race" is several agents taking turns on one *physically shared* board
 * (see `rl/evaluation/shared_race.py`) -- not independent episodes matched
 * by seed. Reuses `ReplayAction` from `types/replay.ts` for a turn's action.
 */

import type { ReplayAction } from "@/types/replay";

export interface RaceTurn {
  turn: number;
  agent: string;
  action: ReplayAction;
  /** The *shared* board immediately after this turn. Still `-1` at this
   * turn's own cell when `eliminated` is true -- a fatal cell is never
   * written into the board, only ever derivable from `action` + `eliminated`
   * (same convention `ReplayDetail` already uses for a lost solo episode). */
  board_state: number[][];
  eliminated: boolean;
  reasoning: Record<string, unknown> | null;
}

export interface RaceSummary {
  id: string;
  seed: number;
  board_size: string;
  mines: number;
  /** Fixed round-robin turn order, e.g. ["Random", "CSP", "DQN", "PPO"]. */
  turn_order: string[];
  /** Whether the board was collectively cleared before every agent was eliminated. */
  won: boolean;
  total_turns: number;
  generated_at: string | null;
}

export interface RaceDetail {
  id: string;
  seed: number;
  board_size: string;
  mines: number;
  turn_order: string[];
  generated_at: string | null;
  /** The all-hidden board before any agent acted. */
  initial_board: number[][];
  turns: RaceTurn[];
  won: boolean;
  total_turns: number;
  surviving_agents: string[];
  /** Agent name -> the turn number that eliminated them. */
  eliminated_agents: Record<string, number>;
}
