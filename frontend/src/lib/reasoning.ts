/**
 * A single human-readable "why" sentence for one replay step's recorded
 * reasoning -- used by the Home page's replay showcase preview card. Mirrors
 * the same per-agent shapes `ReplayInfo` and `AgentConfigShowcase` render in
 * full detail elsewhere (`ReplayStep.reasoning`, see `types/replay.ts`), just
 * condensed to one line for a preview. Never fabricates a reason when
 * nothing was recorded for that step.
 *
 * "Highest Q-value"/"Highest action probability" are accurate, not invented
 * phrasing: replays are generated from each agent's greedy (non-exploring)
 * action selection, so the recorded value for the chosen cell genuinely was
 * the highest among the agent's options at that step -- see
 * `QLearningAgent.select_action`/`DQNAgent`'s eval-mode `choose_action` in
 * `rl/agents/`.
 */
export function summarizeReason(agent: string, reasoning: Record<string, unknown> | null): string | null {
  if (!reasoning) return null;

  if (agent === "DQN" && typeof reasoning.q_value === "number") {
    return `Highest Q-value (${reasoning.q_value.toFixed(3)})`;
  }
  if (agent === "PPO" && typeof reasoning.action_probability === "number") {
    return `Highest action probability (${(reasoning.action_probability * 100).toFixed(1)}%)`;
  }
  if (agent === "CSP" && typeof reasoning.inference === "string") {
    return reasoning.inference;
  }
  return null;
}

/**
 * Extends `summarizeReason` with one honest static fallback: Random records
 * no per-step reasoning at all (`ReplayStep.reasoning` is always `null` for
 * it -- there's nothing to compute a reason from), but *why* is itself a
 * fixed, well-known fact about the agent -- see `RandomAgent`/
 * `AGENT_EXPLAINERS.random` -- worth stating plainly rather than leaving the
 * "Reason" field blank next to every other agent's.
 */
export function describeDecisionReason(agent: string, reasoning: Record<string, unknown> | null): string | null {
  const summarized = summarizeReason(agent, reasoning);
  if (summarized) return summarized;
  if (agent === "Random") return "Uniform random selection -- no learning involved.";
  return null;
}
