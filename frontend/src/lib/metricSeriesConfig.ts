/**
 * Which chart(s) to render per agent, and which history field feeds each
 * one. DQN and PPO log different diagnostics (see
 * `rl/agents/dqn_agent.py`/`rl/agents/ppo_agent.py`'s `train()` return
 * shape), so this is agent-specific rather than one fixed chart list.
 */
export interface MetricSeriesConfig {
  key: string;
  label: string;
}

export const METRIC_SERIES_BY_AGENT: Record<string, MetricSeriesConfig[]> = {
  DQN: [
    { key: "reward_rolling_mean", label: "Reward (rolling mean)" },
    { key: "loss", label: "Loss" },
    { key: "epsilon", label: "Epsilon" },
    { key: "win_rate_rolling", label: "Win rate (rolling)" },
    { key: "avg_q", label: "Avg. Q-value" },
    { key: "max_q", label: "Max Q-value" },
  ],
  PPO: [
    { key: "total_reward", label: "Reward" },
    { key: "policy_loss", label: "Policy loss" },
    { key: "value_loss", label: "Value loss" },
    { key: "entropy", label: "Entropy" },
    { key: "explained_variance", label: "Explained variance" },
  ],
};

/** Only chart series the response actually reports (via its `series` field) --
 * defends against a future agent/schema this config hasn't been updated for. */
export function resolveChartableSeries(agent: string, availableSeries: string[]): MetricSeriesConfig[] {
  const configured = METRIC_SERIES_BY_AGENT[agent] ?? [];
  return configured.filter((series) => availableSeries.includes(series.key));
}
