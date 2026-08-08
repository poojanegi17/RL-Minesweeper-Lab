import { getBoardConfigs } from "@/api/boardConfigs";
import { getLeaderboard } from "@/api/metrics";
import type { LeaderboardEntry } from "@/types/metrics";

export const LEVEL_ORDER = ["beginner", "intermediate", "expert"];
export const DENSITY_ORDER = ["sparse", "standard", "dense"];

export interface BoardConfigSnapshot {
  level: string;
  density: string;
  rows: number;
  cols: number;
  mines: number;
  entries: LeaderboardEntry[];
}

/**
 * Every (level, density) leaderboard snapshot the board-config catalog
 * defines, fetched in parallel -- one `GET /api/leaderboard` call per
 * config, each already covering every agent, so this is 9 calls total
 * rather than 9 calls per agent. Shared by `BoardConfigComparisonTable`
 * (per-agent detail rows in the research chamber) and `ResearchPipeline`
 * (each flow card's best-seen-so-far win rate).
 */
export async function fetchAllBoardConfigLeaderboards(): Promise<BoardConfigSnapshot[]> {
  const configs = await getBoardConfigs();
  const requests: { level: string; density: string; rows: number; cols: number; mines: number }[] = [];

  for (const level of LEVEL_ORDER) {
    const config = configs.find((c) => c.level === level);
    if (!config) continue;
    for (const density of DENSITY_ORDER) {
      const mines = config.densities[density];
      if (mines == null) continue;
      requests.push({ level, density, rows: config.rows, cols: config.cols, mines });
    }
  }

  return Promise.all(requests.map(async (request) => ({ ...request, entries: await getLeaderboard(request.level, request.density) })));
}

/** Best win rate seen for `agent` across every fetched snapshot, or `null` if
 * none of them have a recorded win rate for it yet. */
export function bestWinRateFor(agent: string, snapshots: BoardConfigSnapshot[]): number | null {
  let best: number | null = null;
  for (const snapshot of snapshots) {
    const winRate = snapshot.entries.find((entry) => entry.agent === agent)?.win_rate;
    if (winRate == null) continue;
    if (best == null || winRate > best) best = winRate;
  }
  return best;
}
