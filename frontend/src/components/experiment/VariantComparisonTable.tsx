import { GitBranch } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { Badge } from "@/components/ui/Badge";
import { PlaceholderChart } from "@/components/charts/PlaceholderChart";
import { useTheme } from "@/app/ThemeProvider";
import { agentKindFromName } from "@/lib/agentAdapters";
import { formatPercent, formatReward, humanizeVariant } from "@/lib/experimentAdapters";
import { AGENT_HEX } from "@/data/types";
import type { RunBrief } from "@/types/experiment";

interface VariantComparisonTableProps {
  /** The family's runs, straight from `ExperimentSummary.runs` -- no separate
   * `/ablation` request. Grouping is already done server-side (the same
   * ablation-style id pattern), so there's nothing left to re-derive here. */
  runs: RunBrief[];
  /** The run currently being viewed, highlighted in the table -- absent when
   * this is rendered from the family's own page (nothing to highlight). */
  currentRunId?: string;
  /** Needed to color the bar chart -- runs share one agent, but `RunBrief`
   * doesn't repeat it per run, so the caller passes it. */
  agent: string;
}

/**
 * Variant comparison for a family of related runs -- baseline vs.
 * +checkpoint vs. +LR decay vs. combined, or any other family following the
 * backend's generic `<prefix>_<LETTER>_<variant>` id convention. Renders
 * directly from the grouped `runs` array a family (or an individual run's
 * `family_id` lookup) already carries, rather than a second endpoint.
 *
 * Shows an empty state when there's nothing to compare (a single run).
 */
export function VariantComparisonTable({ runs, currentRunId, agent }: VariantComparisonTableProps) {
  const { theme } = useTheme();

  if (runs.length <= 1) {
    return (
      <EmptyState
        icon={GitBranch}
        title="No variants to compare"
        description="This experiment has no sibling runs -- it's a standalone run, not part of a family."
      />
    );
  }

  const color = AGENT_HEX[agentKindFromName(agent)][theme];
  const chartData = runs
    .filter((run) => run.win_rate != null)
    .map((run) => ({
      label: run.variant ? humanizeVariant(run.variant) : run.title,
      value: Math.round(run.win_rate! * 1000) / 10,
      color: run.id === currentRunId ? color : `${color}80`,
    }));

  return (
    <div className="flex flex-col gap-6">
      {chartData.length > 0 && (
        <Card>
          <h3 className="mb-4 text-xs font-medium tracking-wide text-text-muted uppercase">
            Win rate by variant
          </h3>
          <PlaceholderChart data={chartData} color={color} />
        </Card>
      )}

      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-border text-left text-xs text-text-muted uppercase">
              <th className="pb-2 pr-3 font-medium">Variant</th>
              <th className="pb-2 pr-3 font-medium">Run</th>
              <th className="pb-2 pr-3 font-medium">Episodes</th>
              <th className="pb-2 pr-3 font-medium">Win rate</th>
              <th className="pb-2 font-medium">Avg. reward</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr
                key={run.id}
                className="border-b border-border last:border-0"
                aria-current={run.id === currentRunId ? "true" : undefined}
              >
                <td className="py-2.5 pr-3">
                  <Badge variant={run.id === currentRunId ? "neutral" : "outline"}>
                    {run.variant ? humanizeVariant(run.variant) : "—"}
                  </Badge>
                </td>
                <td className="py-2.5 pr-3">
                  <span className="font-medium text-heading">{run.title}</span>
                  <code className="ml-2 font-mono text-xs text-text-muted">{run.id}</code>
                </td>
                <td className="py-2.5 pr-3 font-mono">{run.episodes.toLocaleString()}</td>
                <td className="py-2.5 pr-3 font-mono">{formatPercent(run.win_rate)}</td>
                <td className="py-2.5 font-mono">{formatReward(run.avg_reward)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
