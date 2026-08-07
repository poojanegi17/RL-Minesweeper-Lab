import { apiGet } from "@/api/client";
import { levelDensityQuery } from "@/lib/boardLevelQuery";
import type { LeaderboardEntry, MetricsResponse } from "@/types/metrics";

/** GET /api/experiments/{id}/metrics -- full per-episode training history, chart-ready. */
export function getExperimentMetrics(id: string): Promise<MetricsResponse> {
  return apiGet<MetricsResponse>(`/api/experiments/${encodeURIComponent(id)}/metrics`);
}

/** GET /api/leaderboard -- every agent ranked by best known win rate, at the
 * given level/density (see `GET /api/board-configs`); omitted means the
 * default beginner/standard board. */
export function getLeaderboard(level?: string, density?: string): Promise<LeaderboardEntry[]> {
  return apiGet<LeaderboardEntry[]>(`/api/leaderboard${levelDensityQuery(level, density)}`);
}
