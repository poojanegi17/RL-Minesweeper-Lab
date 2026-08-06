import { apiGet } from "@/api/client";
import type { ReplayDetail, ReplaySummary } from "@/types/replay";

/** GET /api/replays -- every replay discoverable under rl/results/replays/. */
export function getReplays(): Promise<ReplaySummary[]> {
  return apiGet<ReplaySummary[]>("/api/replays");
}

/** GET /api/replays/{id} -- full step-by-step timeline for one replay. */
export function getReplay(id: string): Promise<ReplayDetail> {
  return apiGet<ReplayDetail>(`/api/replays/${encodeURIComponent(id)}`);
}
