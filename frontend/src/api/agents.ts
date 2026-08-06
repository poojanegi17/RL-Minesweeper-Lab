import { apiGet } from "@/api/client";
import type { Agent } from "@/types/agent";

/** GET /api/agents -- the static catalog of every agent implemented in this project. */
export function getAgents(): Promise<Agent[]> {
  return apiGet<Agent[]>("/api/agents");
}
