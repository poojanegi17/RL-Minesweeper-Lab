import { apiGet } from "@/api/client";
import type { RaceDetail, RaceSummary } from "@/types/race";

/** GET /api/races -- every race discoverable under rl/results/races/. */
export function getRaces(): Promise<RaceSummary[]> {
  return apiGet<RaceSummary[]>("/api/races");
}

/** GET /api/races/{id} -- full per-agent timelines for one race. */
export function getRace(id: string): Promise<RaceDetail> {
  return apiGet<RaceDetail>(`/api/races/${encodeURIComponent(id)}`);
}
