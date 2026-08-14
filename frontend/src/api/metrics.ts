import { apiGet } from "@/api/client";
import { levelDensityQuery, type FirstClickPolicy } from "@/lib/boardLevelQuery";
import type { LeaderboardEntry, MetricsResponse } from "@/types/metrics";

/** GET /api/experiments/{id}/metrics -- full per-episode training history, chart-ready. */
export function getExperimentMetrics(id: string): Promise<MetricsResponse> {
  return apiGet<MetricsResponse>(`/api/experiments/${encodeURIComponent(id)}/metrics`);
}

/** GET /api/leaderboard -- every agent ranked by best known win rate, at the
 * given level/density (see `GET /api/board-configs`); omitted means the
 * default beginner/standard board.
 *
 * `firstClickSafe` selects one board distribution's own results tree. Omit it
 * for the default tree, which is what every caller that isn't explicitly
 * comparing the two distributions should do. */
export function getLeaderboard(level?: string, density?: string, firstClickSafe?: FirstClickPolicy): Promise<LeaderboardEntry[]> {
  return apiGet<LeaderboardEntry[]>(`/api/leaderboard${levelDensityQuery(level, density, firstClickSafe)}`);
}
