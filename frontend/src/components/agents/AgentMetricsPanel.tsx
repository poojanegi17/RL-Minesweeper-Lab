import { Hourglass } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { EmptyState } from "@/components/ui/EmptyState";
import { ExperimentMetricsChart } from "@/components/charts/ExperimentMetricsChart";
import { formatPercent, formatReward } from "@/lib/experimentAdapters";
import type { ExperimentDetail } from "@/types/experiment";
import type { LeaderboardEntry } from "@/types/metrics";
import type { MetricsResponse } from "@/types/metrics";

interface AgentMetricsPanelProps {
  /** Full run detail (hyperparameters aside) when this agent has a trained
   * experiment on the leaderboard -- null for CSP/Q-Learning/Random, which
   * write no experiment artifacts at all. */
  experiment: ExperimentDetail | null;
  /** Per-episode training history for `experiment` -- null in lockstep with it. */
  metrics: MetricsResponse | null;
  leaderboardEntry: LeaderboardEntry | undefined;
}

/**
 * Real metrics only -- a stats row plus `ExperimentMetricsChart`'s
 * agent-aware training curves when a live experiment exists (DQN, PPO), or a
 * reference-figure stat row when the agent only has the project's
 * static-reference leaderboard entry (CSP, Q-Learning, Random -- see
 * `routes/metrics.py`'s module docstring on the backend for why those three
 * have no live artifacts to chart).
 */
export function AgentMetricsPanel({ experiment, metrics, leaderboardEntry }: AgentMetricsPanelProps) {
  if (experiment && metrics) {
    const stats = [
      { label: "Win rate", value: formatPercent(experiment.evaluation_metrics.win_rate) },
      { label: "Avg. reward", value: formatReward(experiment.evaluation_metrics.avg_reward) },
      { label: "Avg. episode length", value: formatReward(experiment.evaluation_metrics.avg_episode_length) },
      { label: "Episodes trained", value: experiment.episodes.toLocaleString() },
    ];

    return (
      <div className="flex flex-col gap-6">
        <Card>
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-medium tracking-wide text-text-muted uppercase">Live training metrics</h3>
            <Badge variant="outline">Live</Badge>
          </div>
          <dl className="mt-4 grid grid-cols-2 gap-6 sm:grid-cols-4">
            {stats.map((stat) => (
              <div key={stat.label}>
                <dd className="font-mono text-xl font-semibold text-heading">{stat.value}</dd>
                <dt className="mt-1 text-xs text-text-muted">{stat.label}</dt>
              </div>
            ))}
          </dl>
        </Card>
        <ExperimentMetricsChart metrics={metrics} />
      </div>
    );
  }

  if (leaderboardEntry?.source === "not_trained") {
    return (
      <EmptyState
        icon={Hourglass}
        title="Not trained at this level yet"
        description="No checkpoint has been trained for this agent at this board size/density yet."
      />
    );
  }

  if (leaderboardEntry) {
    const isBoardResult = leaderboardEntry.source === "board_result";
    const stats = [
      { label: "Win rate", value: formatPercent(leaderboardEntry.win_rate) },
      { label: "Avg. episode length", value: formatReward(leaderboardEntry.avg_episode_length) },
      ...(leaderboardEntry.avg_reward != null ? [{ label: "Avg. reward", value: formatReward(leaderboardEntry.avg_reward) }] : []),
    ];

    return (
      <Card className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-medium tracking-wide text-text-muted uppercase">
            {isBoardResult ? "Evaluation figures" : "Reference figures"}
          </h3>
          <Badge variant="outline">{isBoardResult ? "Evaluated" : "Reference"}</Badge>
        </div>
        <dl className="grid grid-cols-2 gap-6 sm:w-1/2">
          {stats.map((stat) => (
            <div key={stat.label}>
              <dd className="font-mono text-xl font-semibold text-heading">{stat.value}</dd>
              <dt className="mt-1 text-xs text-text-muted">{stat.label}</dt>
            </div>
          ))}
        </dl>
        <p className="text-xs text-text-muted">
          {isBoardResult
            ? "This agent writes no per-episode training history at this board size -- these are real evaluation figures, just not a live training chart."
            : "This agent writes no per-episode training history -- these are the project's last-recorded figures, not a live chart."}
        </p>
      </Card>
    );
  }

  return (
    <EmptyState
      icon={Hourglass}
      title="No metrics recorded"
      description="This agent has no recorded results yet."
    />
  );
}
