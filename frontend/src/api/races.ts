import { apiGet } from "@/api/client";
import { levelDensityQuery } from "@/lib/boardLevelQuery";
import type { RaceDetail, RaceSummary } from "@/types/race";

/** GET /api/races -- every race discoverable under rl/results/races/ (or a
 * level/density subdirectory -- see `GET /api/board-configs`). */
export function getRaces(level?: string, density?: string): Promise<RaceSummary[]> {
  return apiGet<RaceSummary[]>(`/api/races${levelDensityQuery(level, density)}`);
}

/** GET /api/races/{id} -- full per-agent timelines for one race. */
export function getRace(id: string, level?: string, density?: string): Promise<RaceDetail> {
  return apiGet<RaceDetail>(`/api/races/${encodeURIComponent(id)}${levelDensityQuery(level, density)}`);
}
