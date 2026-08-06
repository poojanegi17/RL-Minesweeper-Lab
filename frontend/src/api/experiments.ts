import { apiGet } from "@/api/client";
import type { AblationResponse, ExperimentDetail, ExperimentSummary } from "@/types/experiment";

/** GET /api/experiments -- every experiment discoverable under rl/results/. */
export function getExperiments(): Promise<ExperimentSummary[]> {
  return apiGet<ExperimentSummary[]>("/api/experiments");
}

/** GET /api/experiments/{id} -- `id` is tried as a family id first (returning
 * the same grouped `ExperimentSummary` shape, with its full `runs` list);
 * if no family matches, it's treated as an individual run id and full
 * `ExperimentDetail` is returned instead. Use `isFamilySummary` (`@/lib/experimentAdapters`)
 * to tell the two apart. */
export function getExperiment(id: string): Promise<ExperimentSummary | ExperimentDetail> {
  return apiGet<ExperimentSummary | ExperimentDetail>(`/api/experiments/${encodeURIComponent(id)}`);
}

/** GET /api/experiments/{id}/ablation -- sibling experiments sharing an ablation
 * family, for comparison (e.g. baseline vs. +checkpoint vs. +LR decay vs. combined). */
export function getExperimentAblation(id: string): Promise<AblationResponse> {
  return apiGet<AblationResponse>(`/api/experiments/${encodeURIComponent(id)}/ablation`);
}
