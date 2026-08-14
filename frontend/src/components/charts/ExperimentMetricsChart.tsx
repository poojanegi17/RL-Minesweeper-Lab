import { Hourglass } from "lucide-react";
import { EmptyState } from "@/components/ui/EmptyState";
import { MetricLineChart } from "@/components/charts/MetricLineChart";
import { resolveChartableSeries } from "@/lib/metricSeriesConfig";
import type { MetricsResponse } from "@/types/metrics";

interface ExperimentMetricsChartProps {
  metrics: MetricsResponse;
}

/**
 * A grid of training-metric charts for one experiment, agent-aware: DQN and
 * PPO log different diagnostics (`resolveChartableSeries`/`metricSeriesConfig.ts`
 * decide which ones), so only series the response actually reports are
 * rendered -- see `MetricLineChart` for the per-series chart itself.
 *
 * Shared by `AgentDetail` (one representative experiment per agent) and
 * `ExperimentDetail` (any experiment by id) so both pages chart training
 * history identically rather than duplicating this grid.
 */
export function ExperimentMetricsChart({ metrics }: ExperimentMetricsChartProps) {
  const series = resolveChartableSeries(metrics.agent, metrics.series);

  if (series.length === 0) {
    return (
      <EmptyState
        icon={Hourglass}
        title="No chartable metrics"
        description="This experiment's history doesn't include any of the metrics this page knows how to chart."
      />
    );
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {series.map((s) => (
        <div key={s.key} className="rounded-xl border border-border p-4">
          <h3 className="mb-2 text-xs font-medium tracking-wide text-text-muted uppercase">{s.label}</h3>
          <MetricLineChart data={metrics.episodes} metricKey={s.key} />
          <p className="mt-2 text-xs text-text-muted">{s.description}</p>
        </div>
      ))}
    </div>
  );
}
