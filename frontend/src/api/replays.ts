import { apiGet } from "@/api/client";
import { levelDensityQuery, type FirstClickPolicy } from "@/lib/boardLevelQuery";
import type { ReplayDetail, ReplaySummary } from "@/types/replay";

/** GET /api/replays -- every replay discoverable under rl/results/replays/
 * (or a level/density subdirectory -- see `GET /api/board-configs`). */
export function getReplays(level?: string, density?: string, firstClickSafe?: FirstClickPolicy): Promise<ReplaySummary[]> {
  return apiGet<ReplaySummary[]>(`/api/replays${levelDensityQuery(level, density, firstClickSafe)}`);
}

/** GET /api/replays/{id} -- full step-by-step timeline for one replay. */
export function getReplay(
  id: string,
  level?: string,
  density?: string,
  firstClickSafe?: FirstClickPolicy,
): Promise<ReplayDetail> {
  return apiGet<ReplayDetail>(
    `/api/replays/${encodeURIComponent(id)}${levelDensityQuery(level, density, firstClickSafe)}`,
  );
}
