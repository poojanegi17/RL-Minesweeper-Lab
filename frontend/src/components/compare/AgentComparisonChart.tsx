import { useMemo } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Hourglass } from "lucide-react";
import { EmptyState } from "@/components/ui/EmptyState";
import { downsample } from "@/lib/downsample";
import { computeRollingWinRate } from "@/lib/rollingWinRate";
import type { AgentKind } from "@/data/types";
import type { MetricsResponse } from "@/types/metrics";

const MAX_CHART_POINTS = 400;
const ROLLING_WINDOW = 200;

interface ComparisonSeries {
  name: string;
  kind: AgentKind;
  metrics: MetricsResponse | null;
}

interface AgentComparisonChartProps {
  agentA: ComparisonSeries;
  agentB: ComparisonSeries;
  colorA: string;
  colorB: string;
}

/**
 * Overlays up to two agents' rolling win rate on one chart, on a shared
 * numeric episode axis -- each `<Line>` carries its own `data` (Recharts
 * supports this per-line, needed here since DQN/PPO training runs rarely
 * have the same episode count). Only agents with `metrics` (DQN/PPO today --
 * Random/CSP/Q-Learning write no per-episode history, see
 * `services/results_loader.py`'s module docstring) contribute a line.
 */
export function AgentComparisonChart({ agentA, agentB, colorA, colorB }: AgentComparisonChartProps) {
  const seriesA = useMemo(
    () => (agentA.metrics ? downsample(computeRollingWinRate(agentA.metrics.episodes, ROLLING_WINDOW), MAX_CHART_POINTS) : null),
    [agentA.metrics],
  );
  const seriesB = useMemo(
    () => (agentB.metrics ? downsample(computeRollingWinRate(agentB.metrics.episodes, ROLLING_WINDOW), MAX_CHART_POINTS) : null),
    [agentB.metrics],
  );

  if (!seriesA && !seriesB) {
    return (
      <EmptyState
        icon={Hourglass}
        title="No training history to compare"
        description="Neither agent recorded a per-episode training history -- only DQN and PPO write one; Random, CSP, and Q-Learning don't train a checkpointed model."
      />
    );
  }

  const maxEpisode = Math.max(seriesA?.at(-1)?.episode ?? 0, seriesB?.at(-1)?.episode ?? 0);

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" opacity={0.6} vertical={false} />
        <XAxis
          dataKey="episode"
          type="number"
          domain={[0, maxEpisode]}
          stroke="var(--color-text-muted)"
          fontSize={11}
          fontFamily="var(--font-mono)"
          tickLine={false}
          axisLine={{ stroke: "var(--color-border)" }}
          minTickGap={40}
        />
        <YAxis
          domain={[0, "auto"]}
          tickFormatter={(value: number) => `${(value * 100).toFixed(0)}%`}
          stroke="var(--color-text-muted)"
          fontSize={11}
          fontFamily="var(--font-mono)"
          tickLine={false}
          axisLine={false}
          width={44}
        />
        <Tooltip
          formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`}
          labelFormatter={(episode) => `Episode ${episode}`}
          cursor={{ stroke: "var(--color-text-muted)", strokeOpacity: 0.3 }}
          contentStyle={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: 8,
            fontSize: 12,
            fontFamily: "var(--font-mono)",
          }}
          labelStyle={{ color: "var(--color-heading)", marginBottom: 2 }}
        />
        <Legend wrapperStyle={{ fontSize: 12, fontFamily: "var(--font-mono)" }} />
        {seriesA && (
          <Line data={seriesA} dataKey="value" name={agentA.name} stroke={colorA} strokeWidth={1.75} dot={false} isAnimationActive={false} />
        )}
        {seriesB && (
          <Line data={seriesB} dataKey="value" name={agentB.name} stroke={colorB} strokeWidth={1.75} dot={false} isAnimationActive={false} />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}
