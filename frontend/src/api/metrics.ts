import { apiGet } from "@/api/client";
import type { LeaderboardEntry, MetricsResponse } from "@/types/metrics";

/** GET /api/experiments/{id}/metrics -- full per-episode training history, chart-ready. */
export function getExperimentMetrics(id: string): Promise<MetricsResponse> {
  return apiGet<MetricsResponse>(`/api/experiments/${encodeURIComponent(id)}/metrics`);
}

/** GET /api/leaderboard -- every agent ranked by best known win rate. */
export function getLeaderboard(): Promise<LeaderboardEntry[]> {
  return apiGet<LeaderboardEntry[]>("/api/leaderboard");
}
