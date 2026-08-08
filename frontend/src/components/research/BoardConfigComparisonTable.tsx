import { useMemo } from "react";
import { Trophy } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { LEVEL_LABELS, DENSITY_LABELS } from "@/components/board/LevelDensitySelector";
import { fetchAllBoardConfigLeaderboards } from "@/lib/boardComparison";
import { useApiQuery } from "@/hooks/useApiQuery";
import { formatPercent, formatReward } from "@/lib/experimentAdapters";
import type { AgentKind } from "@/data/types";
import type { LeaderboardEntry } from "@/types/metrics";

interface BoardConfigComparisonTableProps {
  agentName: string;
  kind: AgentKind;
  accentColor: string;
}

interface ConfigRow {
  level: string;
  density: string;
  rows: number;
  cols: number;
  mines: number;
  entry: LeaderboardEntry | undefined;
}

/**
 * Hand-authored, grounded in the actual observed board-result trend for that
 * agent kind (`evaluate_board_config.py`'s output, see `fetchAllBoardConfigLeaderboards`)
 * -- not a generic auto-derived claim, since a couple of data points aren't
 * enough to safely generalize for every algorithm. Only filled in where the
 * full level x density grid actually supports the claim (Random, CSP have
 * real results at every board size); other kinds fall back to a plain
 * "not enough data yet" note computed from what's actually on the leaderboard.
 */
const TREND_NOTES: Partial<Record<AgentKind, string>> = {
  random:
    "Random has no state and no deduction, so a bigger board never helps it: win rate falls as mine density rises (more mines packed into the same hidden-cell count means a blind guess is more likely to hit one), and it's near zero everywhere except the smallest, sparsest board.",
  "rule-based":
    "CSP's deduction rules can only prove a cell safe when the local constraint arithmetic resolves exactly. Sparser boards leave more of the board provably safe, so win rate is highest at every level's sparse preset and drops as density rises and CSP falls back to a probability guess more often -- true at every board size tested.",
};

async function fetchAgentBoardRows(agentName: string): Promise<ConfigRow[]> {
  const snapshots = await fetchAllBoardConfigLeaderboards();
  return snapshots.map((snapshot) => ({
    level: snapshot.level,
    density: snapshot.density,
    rows: snapshot.rows,
    cols: snapshot.cols,
    mines: snapshot.mines,
    entry: snapshot.entries.find((entry) => entry.agent === agentName),
  }));
}

/**
 * Every board size/density this agent has been evaluated at, side by side --
 * replaces the old per-chamber level/density toggle (which only ever showed
 * one config's numbers at a time) with the full comparison at once, closing
 * with which config actually performed best.
 */
export function BoardConfigComparisonTable({ agentName, kind, accentColor }: BoardConfigComparisonTableProps) {
  const { data: rows, status, error, isSlow, retry } = useApiQuery(() => fetchAgentBoardRows(agentName), [agentName]);

  const { bestRow, levelsWithData } = useMemo(() => {
    let best: ConfigRow | null = null;
    const levels = new Set<string>();
    for (const row of rows ?? []) {
      if (row.entry?.win_rate == null) continue;
      levels.add(row.level);
      if (best?.entry?.win_rate == null || row.entry.win_rate > best.entry.win_rate) best = row;
    }
    return { bestRow: best, levelsWithData: levels };
  }, [rows]);

  if (status === "loading") {
    return (
      <div className="flex flex-col gap-3">
        <Skeleton className="h-56 w-full" />
        {isSlow && <ColdStartNotice />}
      </div>
    );
  }
  if (status === "error" && error) {
    return <ApiErrorState error={error} onRetry={retry} title="Couldn't load board-size results" />;
  }
  if (!rows) return null;

  const trendNote =
    TREND_NOTES[kind] ??
    (levelsWithData.size <= 1
      ? "Only evaluated at one board size so far -- not enough data yet to say how results change with board size or density."
      : null);

  return (
    <div className="flex flex-col gap-3">
      <h4 className="text-xs font-medium tracking-wide text-text-muted uppercase">Results across board size &amp; mine density</h4>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-border text-left text-xs text-text-muted uppercase">
              <th className="pb-2 pr-3 font-medium">Board size</th>
              <th className="pb-2 pr-3 font-medium">Density</th>
              <th className="pb-2 pr-3 font-medium">Mines</th>
              <th className="pb-2 pr-3 font-medium">Win rate</th>
              <th className="pb-2 font-medium">Avg. reward</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const isBest = bestRow?.level === row.level && bestRow?.density === row.density;
              const notTrained = row.entry?.win_rate == null;

              return (
                <tr key={`${row.level}-${row.density}`} className="border-b border-border last:border-0" style={isBest ? { backgroundColor: `${accentColor}0d` } : undefined}>
                  <td className="py-2.5 pr-3">
                    <span className="font-medium text-heading">{LEVEL_LABELS[row.level] ?? row.level}</span>
                    <span className="ml-2 font-mono text-xs text-text-muted">
                      {row.rows}x{row.cols}
                    </span>
                  </td>
                  <td className="py-2.5 pr-3 text-text">{DENSITY_LABELS[row.density] ?? row.density}</td>
                  <td className="py-2.5 pr-3 font-mono">{row.mines}</td>
                  <td className="py-2.5 pr-3">
                    {notTrained ? (
                      <Badge variant="outline">Not trained</Badge>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 font-mono font-medium" style={isBest ? { color: accentColor } : undefined}>
                        {formatPercent(row.entry!.win_rate)}
                        {isBest && <Trophy className="h-3 w-3" style={{ color: accentColor }} aria-hidden="true" />}
                      </span>
                    )}
                  </td>
                  <td className="py-2.5 font-mono">{row.entry?.avg_reward != null ? formatReward(row.entry.avg_reward) : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {trendNote && <p className="text-sm text-text-muted">{trendNote}</p>}

      <p className="text-sm font-medium text-heading">
        {bestRow && bestRow.entry ? (
          <>
            Best performing configuration:{" "}
            <span style={{ color: accentColor }}>
              {LEVEL_LABELS[bestRow.level] ?? bestRow.level} / {DENSITY_LABELS[bestRow.density] ?? bestRow.density}
            </span>{" "}
            ({bestRow.rows}x{bestRow.cols}, {bestRow.mines} mines) at {formatPercent(bestRow.entry.win_rate)} win rate.
          </>
        ) : (
          "No board-size evaluation recorded for this agent yet."
        )}
      </p>
    </div>
  );
}
