import type { AgentKind, AgentStatus, Agent as UiAgent, Metric } from "@/data/types";
import type { Agent as ApiAgent } from "@/types/agent";
import type { LeaderboardEntry } from "@/types/metrics";

/**
 * Adapts the backend's `AgentInfo` shape (`@/types/agent`) into the richer
 * UI-facing `Agent` shape (`@/data/types`) that `AgentCard` and `AgentDetail`
 * were already built around -- so those components and the validated agent
 * color palette don't need to change, just their data source.
 */

const NAME_TO_KIND: Record<string, AgentKind> = {
  Random: "random",
  CSP: "rule-based",
  "Q-Learning": "q-learning",
  DQN: "dqn",
  PPO: "ppo",
};

/** URL-safe id derived from the API's agent name (e.g. "Q-Learning" -> "q-learning"). */
export function slugifyAgentName(name: string): string {
  return name.toLowerCase().replace(/\s+/g, "-");
}

/** Falls back to "random"'s neutral styling for any future agent name this
 * mapping hasn't been updated for, rather than throwing. */
export function agentKindFromName(name: string): AgentKind {
  return NAME_TO_KIND[name] ?? "random";
}

export function toUiAgent(apiAgent: ApiAgent, leaderboardEntry?: LeaderboardEntry): UiAgent {
  const status: AgentStatus = apiAgent.has_experiment_artifacts ? "trained" : "baseline";

  const metrics: Metric[] = [];
  if (leaderboardEntry?.win_rate != null) {
    metrics.push({ label: "Win Rate", value: Math.round(leaderboardEntry.win_rate * 1000) / 10, unit: "%" });
  }
  if (leaderboardEntry?.avg_episode_length != null) {
    metrics.push({
      label: "Avg. Episode Length",
      value: Math.round(leaderboardEntry.avg_episode_length * 100) / 100,
    });
  }

  return {
    id: slugifyAgentName(apiAgent.name),
    kind: agentKindFromName(apiAgent.name),
    name: apiAgent.name,
    type: apiAgent.type,
    tagline: apiAgent.description,
    description: apiAgent.description,
    architecture: "",
    status,
    metrics,
  };
}
