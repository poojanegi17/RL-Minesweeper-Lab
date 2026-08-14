import { Trophy } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { DENSITY_LABELS } from "@/components/board/LevelDensitySelector";
import { fetchAllBoardConfigLeaderboards } from "@/lib/boardComparison";
import { useApiQuery } from "@/hooks/useApiQuery";
import { formatPercent, formatReward } from "@/lib/experimentAdapters";

interface DensityResult {
  win_rate: number;
  avg_episode_length: number;
  avg_reward: number;
}

interface BoardSizeDensityTableProps {
  level: string;
  agentName: string;
  /** This run's own recorded density (always "standard" -- every board-size
   * training run in this project trains at that level's standard density)
   * -- the middle row, not a separate measurement, since the run itself
   * *is* that evaluation. */
  standard: DensityResult;
  accentColor: string;
}

const DENSITY_ORDER = ["sparse", "standard", "dense"];

/**
 * This board-size training run's result across all three of that level's
 * mine densities, not just the one it was trained at -- same "does it hold
 * up away from its training config" question `VariantDensityTable` asks per
 * 5x5 variant, asked here per board-size run instead. Sparse/Dense come from
 * `fetchAllBoardConfigLeaderboards` (real `evaluate_board_config.py`-style
 * evaluations of this exact checkpoint, no retraining); Standard is passed
 * in directly since it's this run's own recorded result, not a separate
 * board-result file. Renders nothing until at least one of sparse/dense has
 * real data.
 */
export function BoardSizeDensityTable({ level, agentName, standard, accentColor }: BoardSizeDensityTableProps) {
  const { data: snapshots, status } = useApiQuery(fetchAllBoardConfigLeaderboards, []);

  if (status !== "success" || !snapshots) return null;

  const levelSnapshots = snapshots.filter((s) => s.level === level);
  const results: Record<string, DensityResult | undefined> = { standard };
  let rowsCols: { rows: number; cols: number } | null = null;
  const mines: Record<string, number> = {};

  for (const snapshot of levelSnapshots) {
    mines[snapshot.density] = snapshot.mines;
    if (!rowsCols) rowsCols = { rows: snapshot.rows, cols: snapshot.cols };
    if (snapshot.density === "standard") continue;
    const entry = snapshot.entries.find((e) => e.agent === agentName);
    if (entry?.win_rate != null && entry.avg_episode_length != null && entry.avg_reward != null) {
      results[snapshot.density] = { win_rate: entry.win_rate, avg_episode_length: entry.avg_episode_length, avg_reward: entry.avg_reward };
    }
  }

  const hasAnyGeneralization = results.sparse != null || results.dense != null;
  if (!hasAnyGeneralization) return null;

  const bestDensity = DENSITY_ORDER.reduce<string | null>((best, density) => {
    const result = results[density];
    if (!result) return best;
    if (best === null || result.win_rate > (results[best]?.win_rate ?? -1)) return density;
    return best;
  }, null);

  return (
    <div>
      <p className="mb-1.5 text-xs font-medium tracking-wide text-text-muted uppercase">Result across mine density</p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[420px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-border text-left text-xs text-text-muted uppercase">
              <th className="pb-2 pr-3 font-medium">Density</th>
              <th className="pb-2 pr-3 font-medium">Mines</th>
              <th className="pb-2 pr-3 font-medium">Win rate</th>
              <th className="pb-2 font-medium">Avg. reward</th>
            </tr>
          </thead>
          <tbody>
            {DENSITY_ORDER.map((density) => {
              const result = results[density];
              const isBest = density === bestDensity;
              const isTrainedHere = density === "standard";
              return (
                <tr key={density} className="border-b border-border last:border-0" style={isBest ? { backgroundColor: `${accentColor}0d` } : undefined}>
                  <td className="py-2 pr-3">
                    <span className="text-text">{DENSITY_LABELS[density] ?? density}</span>
                    {isTrainedHere && (
                      <Badge variant="outline" className="ml-2">
                        Trained here
                      </Badge>
                    )}
                  </td>
                  <td className="py-2 pr-3 font-mono">{mines[density] ?? "—"}</td>
                  <td className="py-2 pr-3">
                    {result ? (
                      <span className="inline-flex items-center gap-1.5 font-mono font-medium" style={isBest ? { color: accentColor } : undefined}>
                        {formatPercent(result.win_rate)}
                        {isBest && <Trophy className="h-3 w-3" style={{ color: accentColor }} aria-hidden="true" />}
                      </span>
                    ) : (
                      <Badge variant="outline">Not evaluated</Badge>
                    )}
                  </td>
                  <td className="py-2 font-mono">{result ? formatReward(result.avg_reward) : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-1.5 text-xs text-text-muted">
        Only {rowsCols ? `${rowsCols.rows}x${rowsCols.cols}` : "this level"} Standard is this checkpoint's own training
        density. Sparse and Dense reuse the exact same trained network with no retraining -- a generalization test, not
        a matched-condition result.
      </p>
    </div>
  );
}
