import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { downsample } from "@/lib/downsample";
import type { MetricPoint } from "@/types/metrics";

const MAX_CHART_POINTS = 400;

interface MetricLineChartProps {
  data: MetricPoint[];
  /** Key into each MetricPoint, e.g. "loss", "entropy", "win_rate_rolling". */
  metricKey: string;
  color?: string;
}

/**
 * A single metric plotted against episode number. Styled to match
 * `PlaceholderChart`'s CSS-variable-driven theming (so charts look
 * consistent across the app in both light and dark mode), but using a line
 * rather than bars, since this is a ~thousands-of-points time series rather
 * than a handful of categorical comparisons.
 *
 * Null/undefined values (common before training actually starts producing
 * a given metric -- see `rl/agents/*.py`) are filtered out rather than
 * plotted as zero, which would misrepresent "not yet recorded" as "recorded
 * as zero."
 */
export function MetricLineChart({ data, metricKey, color = "#7c3aed" }: MetricLineChartProps) {
  const points = useMemo(() => {
    const withValue = data
      .filter((row) => typeof row[metricKey] === "number")
      .map((row) => ({ episode: row.episode, value: row[metricKey] as number }));
    return downsample(withValue, MAX_CHART_POINTS);
  }, [data, metricKey]);

  if (points.length === 0) {
    return (
      <div className="flex h-[180px] items-center justify-center text-sm text-text-muted">
        No data recorded for this metric.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={180}>
      <LineChart data={points} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="var(--color-border)"
          opacity={0.6}
          vertical={false}
        />
        <XAxis
          dataKey="episode"
          stroke="var(--color-text-muted)"
          fontSize={11}
          fontFamily="var(--font-mono)"
          tickLine={false}
          axisLine={{ stroke: "var(--color-border)" }}
          minTickGap={40}
        />
        <YAxis
          stroke="var(--color-text-muted)"
          fontSize={11}
          fontFamily="var(--font-mono)"
          tickLine={false}
          axisLine={false}
          width={44}
        />
        <Tooltip
          cursor={{ stroke: "var(--color-text-muted)", strokeOpacity: 0.3 }}
          contentStyle={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: 8,
            fontSize: 12,
            fontFamily: "var(--font-mono)",
          }}
          labelStyle={{ color: "var(--color-heading)", marginBottom: 2 }}
          labelFormatter={(episode) => `Episode ${episode}`}
        />
        <Line
          type="monotone"
          dataKey="value"
          stroke={color}
          strokeWidth={1.75}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
