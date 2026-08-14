import { Trophy } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { DENSITY_LABELS } from "@/components/board/LevelDensitySelector";
import { formatPercent, formatReward } from "@/lib/experimentAdapters";
import { VARIANT_DENSITY_RESULTS, type DensityResult } from "@/lib/variantDensityResults";

interface VariantDensityTableProps {
  /** Real experiment id (`RunBrief.id`) -- keys `VARIANT_DENSITY_RESULTS`. */
  runId: string;
  /** This run's own recorded standard-density (5 mines) result -- the
   * middle row, not a separate measurement (the run itself *is* that
   * evaluation, at 5x5/5 mines). */
  standard: DensityResult;
  accentColor: string;
}

/** 5x5's three density presets, from `board_configs.py` -- fixed here since
 * every DQN/PPO variant in this family trained and evaluated at Beginner. */
const DENSITY_MINES: Record<string, number> = { sparse: 3, standard: 5, dense: 8 };
const DENSITY_ORDER = ["sparse", "standard", "dense"];

/**
 * This one variant's result across all three 5x5 mine densities, not just
 * the one it happened to train at -- the same "does it hold up away from
 * its training config" question `BoardConfigComparisonTable` asks per agent,
 * asked here per variant instead. Only Standard is this checkpoint's own
 * training density; Sparse/Dense are the *same* trained network evaluated on
 * mine counts it never saw during training, with zero retraining -- a
 * generalization test, not a matched-condition result (mirrors
 * `evaluate_board_config.py`'s own DQN/PPO methodology at the agent level).
 * Renders nothing if this run has no recovered sparse/dense data (see
 * `VARIANT_DENSITY_RESULTS`'s docstring).
 */
export function VariantDensityTable({ runId, standard, accentColor }: VariantDensityTableProps) {
  const extra = VARIANT_DENSITY_RESULTS[runId];
  if (!extra) return null;

  const results: Record<string, DensityResult> = { sparse: extra.sparse, standard, dense: extra.dense };
  const bestDensity = DENSITY_ORDER.reduce<string | null>((best, density) => {
    if (best === null || results[density].win_rate > results[best].win_rate) return density;
    return best;
  }, null);

  return (
    <div>
      <p className="mb-1.5 text-xs font-medium tracking-wide text-text-muted uppercase">Result across mine density (5x5)</p>
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
                  <td className="py-2 pr-3 font-mono">{DENSITY_MINES[density]}</td>
                  <td className="py-2 pr-3">
                    <span className="inline-flex items-center gap-1.5 font-mono font-medium" style={isBest ? { color: accentColor } : undefined}>
                      {formatPercent(result.win_rate)}
                      {isBest && <Trophy className="h-3 w-3" style={{ color: accentColor }} aria-hidden="true" />}
                    </span>
                  </td>
                  <td className="py-2 font-mono">{formatReward(result.avg_reward)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-1.5 text-xs text-text-muted">
        Only Standard (5 mines) is this checkpoint's own training density. Sparse and Dense reuse the exact same
        trained network with no retraining -- a generalization test, not a matched-condition result.
      </p>
    </div>
  );
}
