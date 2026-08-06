/**
 * Mirrors `backend/app/schemas/agent.py`'s `AgentInfo`.
 *
 * Deliberately separate from `@/data/types`' `Agent` (the richer UI shape
 * consumed by AgentCard/AgentDetail) -- that type also carries
 * fields the API doesn't provide (tagline, a color/icon `kind`, a headline
 * metrics array). See `@/lib/agentAdapters` for the mapping between the two.
 */
export interface Agent {
  name: string;
  type: string;
  description: string;
  has_experiment_artifacts: boolean;
}
