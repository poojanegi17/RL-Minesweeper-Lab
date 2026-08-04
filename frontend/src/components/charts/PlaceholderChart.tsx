import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface ChartDatum {
  label: string;
  value: number;
  color?: string;
}

interface PlaceholderChartProps {
  data: ChartDatum[];
  color?: string;
}

export function PlaceholderChart({ data, color = "#7c3aed" }: PlaceholderChartProps) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 20, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="var(--color-border)"
          opacity={0.6}
          vertical={false}
        />
        <XAxis
          dataKey="label"
          stroke="var(--color-text-muted)"
          fontSize={12}
          fontFamily="var(--font-mono)"
          tickLine={false}
          axisLine={{ stroke: "var(--color-border)" }}
        />
        <YAxis
          stroke="var(--color-text-muted)"
          fontSize={12}
          fontFamily="var(--font-mono)"
          tickLine={false}
          axisLine={false}
          width={36}
        />
        <Tooltip
          cursor={{ fill: "var(--color-text-muted)", opacity: 0.08 }}
          contentStyle={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: 8,
            fontSize: 12,
            fontFamily: "var(--font-mono)",
          }}
          labelStyle={{ color: "var(--color-heading)", marginBottom: 2 }}
        />
        <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={64}>
          <LabelList
            dataKey="value"
            position="top"
            offset={8}
            fill="var(--color-text-muted)"
            fontSize={11}
            fontFamily="var(--font-mono)"
          />
          {data.map((entry, index) => (
            <Cell key={`${entry.label}-${index}`} fill={entry.color ?? color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
