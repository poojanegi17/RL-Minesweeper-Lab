import { apiGet } from "@/api/client";
import { levelDensityQuery, type FirstClickPolicy } from "@/lib/boardLevelQuery";
import type { RaceDetail, RaceSummary } from "@/types/race";

/** GET /api/races -- every race discoverable under rl/results/races/ (or a
 * level/density subdirectory -- see `GET /api/board-configs`). Passing
 * `firstClickSafe` reads that distribution's own tree: a race where the
 * opening click can lose is a different game from one where it cannot. */
export function getRaces(
  level?: string,
  density?: string,
  firstClickSafe?: FirstClickPolicy,
): Promise<RaceSummary[]> {
  return apiGet<RaceSummary[]>(`/api/races${levelDensityQuery(level, density, firstClickSafe)}`);
}

/** GET /api/races/{id} -- full per-agent timelines for one race. */
export function getRace(
  id: string,
  level?: string,
  density?: string,
  firstClickSafe?: FirstClickPolicy,
): Promise<RaceDetail> {
  return apiGet<RaceDetail>(
    `/api/races/${encodeURIComponent(id)}${levelDensityQuery(level, density, firstClickSafe)}`,
  );
}
