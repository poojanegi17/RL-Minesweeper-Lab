/**
 * Which chart(s) to render per agent, and which history field feeds each
 * one. DQN and PPO log different diagnostics (see
 * `rl/agents/dqn_agent.py`/`rl/agents/ppo_agent.py`'s `train()` return
 * shape), so this is agent-specific rather than one fixed chart list.
 */
export interface MetricSeriesConfig {
  key: string;
  label: string;
  /** One line: what this series actually measures and why it's worth
   * watching -- shown as a caption under its chart. */
  description: string;
}

export const METRIC_SERIES_BY_AGENT: Record<string, MetricSeriesConfig[]> = {
  DQN: [
    {
      key: "reward_rolling_mean",
      label: "Reward (rolling mean)",
      description: "Rolling average of episode reward -- smooths out single-episode noise to show the actual training trend.",
    },
    {
      key: "loss",
      label: "Loss",
      description: "The network's Huber training loss each episode -- spikes here mean the Q-value predictions are diverging, not just noisy.",
    },
    {
      key: "epsilon",
      label: "Epsilon",
      description: "The exploration rate -- starts high (mostly random moves) and decays toward a floor as training progresses, matching the epsilon-greedy schedule.",
    },
    {
      key: "win_rate_rolling",
      label: "Win rate (rolling)",
      description: "Rolling win rate during training -- the training-time signal the eval-time win rate above is trying to predict.",
    },
    {
      key: "avg_q",
      label: "Avg. Q-value",
      description: "Average predicted Q-value across actions -- tracks whether the network's own value estimates are growing unboundedly (a symptom of instability) or settling.",
    },
    {
      key: "max_q",
      label: "Max Q-value",
      description: "The highest predicted Q-value seen -- large or erratic spikes here are often the first visible sign of the overestimation bias Double DQN is meant to control.",
    },
  ],
  PPO: [
    {
      key: "total_reward",
      label: "Reward",
      description: "Per-episode reward -- noisier than a rolling average, but the rawest signal of how the policy is doing episode to episode.",
    },
    {
      key: "policy_loss",
      label: "Policy loss",
      description: "The actor's clipped-surrogate loss -- how much the policy update moved this step, capped by the clipping range.",
    },
    {
      key: "value_loss",
      label: "Value loss",
      description: "The critic's regression loss -- how far its state-value predictions are from the actual returns.",
    },
    {
      key: "entropy",
      label: "Entropy",
      description: "How spread out the action distribution is -- high entropy means more exploration; a fast collapse toward zero means the policy is committing to one action too early.",
    },
    {
      key: "explained_variance",
      label: "Explained variance",
      description: "How much of the return's variance the critic's value predictions actually explain -- low values mean the critic isn't giving the actor a reliable baseline to learn from.",
    },
  ],
};

/** Only chart series the response actually reports (via its `series` field) --
 * defends against a future agent/schema this config hasn't been updated for. */
export function resolveChartableSeries(agent: string, availableSeries: string[]): MetricSeriesConfig[] {
  const configured = METRIC_SERIES_BY_AGENT[agent] ?? [];
  return configured.filter((series) => availableSeries.includes(series.key));
}
