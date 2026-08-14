import { useMemo, useState } from "react";
import { Trophy } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { LEVEL_LABELS, DENSITY_LABELS } from "@/components/board/LevelDensitySelector";
import { fetchAllBoardConfigLeaderboards } from "@/lib/boardComparison";
import {
  FIRST_CLICK_POLICY_BLURBS,
  FIRST_CLICK_POLICY_LABELS,
  type FirstClickPolicy,
} from "@/lib/boardLevelQuery";
import { useApiQuery } from "@/hooks/useApiQuery";
import { formatPercent, formatReward } from "@/lib/experimentAdapters";
import { NarrativeText, type Narrative } from "@/components/research/NarrativeText";
import { cn } from "@/lib/cn";
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
 * enough to safely generalize for every algorithm.
 *
 * Only Random, CSP and Q-Learning have an entry. DQN and PPO deliberately do
 * not: their notes quoted figures that went stale every time a run was
 * re-pointed (the DQN one ended up claiming a best-anywhere of 18.90% when the
 * real figure was 70.35%), and both agents' per-density story is already told
 * by their own pipeline chapters, which read it from the API instead of from a
 * hand-written paragraph. A kind with no entry falls back to a plain "not
 * enough data yet" note, and renders nothing once it has data at 2+ board
 * sizes -- which is the intended state for those two.
 *
 * Figures quoted here are 2,000-episode results under `first_click_safe="area"`
 * (v2, from `rl/evaluation/rebaseline_board_configs.py`); the notes are withheld
 * entirely under the "none" toggle, because they name v2 numbers and a v1 table
 * would contradict them. Q-Learning trains for 100,000 episodes under both
 * distributions, so its toggle compares the environment alone. Supporting
 * structural numbers: CSP's
 * deduction/guess ratios and forced-guess loss rates from
 * `analyze_csp_structure.py` (`rl/analysis/csp_structure_v{1,2}.json`),
 * Q-Learning's state-repetition counts from `analyze_q_learning_coverage.py`,
 * DQN's episode depths from the analysis JSON under `rl/analysis/`. Don't edit
 * a number here without re-running those.
 */
const TREND_NOTES: Partial<Record<AgentKind, Narrative>> = {
  random: [
    "Nothing in this table is strategy. Random has no state and reads nothing, so it measures only how much each board gives away for free.",
    "Down a column (more mines, same board): every blind click is likelier to be fatal, so the win rate falls — 10.20% to 1.30% to 0.15% at Beginner.",
    "Across rows (bigger board): the free opening cascade can no longer cover a whole board, so the wins disappear entirely. Not one win in 2,000 episodes at any Expert density.",
    "Its best cell is the smallest, sparsest one, won in an average of 3.20 clicks — the opening is doing the work, not the agent.",
  ],
  "rule-based": [
    "Two separate forces move these numbers in opposite directions, and this is the only agent here where the board size helps.",
    "Down a column (more mines): win rate falls. Denser boards leave less provably safe, so CSP is pushed into more gambles — true at all three sizes.",
    "Across rows (bigger board): win rate rises. More board gives the constraints more to interlock with, so CSP proves more and guesses about the same.",
    "The two combine to put CSP's peak at Intermediate, not Beginner — 89.95% against 70.35% at standard density. 5x5 is simply too small for constraints to reinforce each other.",
    "Expert does not beat Intermediate because a 16x16 win needs all 216 safe cells cleared, so there are far more chances to meet a forced gamble along the way.",
    "Density wins in the end: Expert/Dense drops to 12.75%, the worst cell on the grid.",
  ],
  "q-learning": [
    "Only one row exists, and that is deliberate — see \"Beginner only\" above for why bigger boards are not run.",
    "Down the row, density is everything, and it breaks abruptly rather than gradually: 91.20% at 3 mines, 71.70% at 5, then 0.60% at 8.",
    "The cause is how deep into a game the table still recognises the board. At 3 mines it has learned nearly a whole game (a median of 19 of 22 cells), at 5 mines most of one (17 of 20), at 8 mines only 12 of 17.",
    "Three extra mines cost it more than any other agent on this grid loses to the same change — density, not board size, is what breaks a tabular method first.",
    "Switch the toggle and the whole row collapses: 58.10% / 1.90% / 0.00% without a safe opening, at the identical 100,000-episode budget. No other agent here depends on that setting so heavily.",
  ],
};

/**
 * Agent kinds whose grid was measured under *both* first-click policies, and
 * so can offer the distribution toggle.
 *
 * Random, CSP and Q-Learning need no checkpoint at any board size, so
 * `rebaseline_board_configs.py` could re-measure their whole grid under
 * `first_click_safe="area"` in one pass while `results_public/v1/levels/`
 * keeps the original `"none"` grid -- both complete, both real.
 *
 * DQN and PPO were excluded for a long time, because a learned agent has to be
 * *retrained* per distribution before it can be re-evaluated. That is no longer
 * a reason: both now have a v2-trained grid of their own (`dqn_v2_A_baseline`,
 * `ppo_v2_J_fully_conv`) alongside their v1 one, so each toggle position is a
 * genuine matched measurement rather than one checkpoint scored on the other's
 * boards.
 *
 * Including them also fixes a mislabelling. A non-toggleable kind left
 * `activePolicy` undefined, which fetched the *default* tree (v1) while the
 * trend note below still rendered under `activePolicy ?? "area"` -- so a v2
 * note could sit under a v1 table with nothing on screen naming either.
 */
const TOGGLEABLE_KINDS = new Set<AgentKind>(["random", "rule-based", "q-learning", "dqn", "ppo"]);

async function fetchAgentBoardRows(agentName: string, firstClickSafe?: FirstClickPolicy): Promise<ConfigRow[]> {
  const snapshots = await fetchAllBoardConfigLeaderboards(firstClickSafe);
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
  const canToggle = TOGGLEABLE_KINDS.has(kind);
  // "area" first: it's the distribution the rest of the current results are
  // measured under, so the default view stays consistent with every other
  // number on the page.
  const [policy, setPolicy] = useState<FirstClickPolicy>("area");
  const activePolicy = canToggle ? policy : undefined;
  const {
    data: rows,
    status,
    error,
    isSlow,
    retry,
  } = useApiQuery(() => fetchAgentBoardRows(agentName, activePolicy), [agentName, activePolicy]);

  // Only show board sizes this agent has actually been evaluated at. A level
  // with no result is dropped rather than rendered as a row of "Not trained"
  // placeholders, which would wrongly imply the run is pending rather than
  // simply not part of this agent's story.
  //
  // This began as a Q-Learning special case (never evaluated beyond Beginner,
  // deliberately -- see TREND_NOTES["q-learning"]) and is now general, because
  // it is not only Q-Learning: neither DQN nor PPO has an Expert run under the
  // current recipe, so both would otherwise show three empty 16x16 rows
  // implying a result that does not exist.
  const visibleRows = useMemo(() => {
    if (!rows) return null;
    const levelsWithEntries = new Set(rows.filter((row) => row.entry?.win_rate != null).map((row) => row.level));
    return rows.filter((row) => levelsWithEntries.has(row.level));
  }, [rows]);

  const { bestRow, levelsWithData } = useMemo(() => {
    let best: ConfigRow | null = null;
    const levels = new Set<string>();
    for (const row of visibleRows ?? []) {
      if (row.entry?.win_rate == null) continue;
      levels.add(row.level);
      if (best?.entry?.win_rate == null || row.entry.win_rate > best.entry.win_rate) best = row;
    }
    return { bestRow: best, levelsWithData: levels };
  }, [visibleRows]);

  // `TREND_NOTES` quote specific figures from the first-click-safe grid, so
  // they are only true under that policy. Under "none" the note is withheld
  // rather than reworded -- an authored paragraph asserting "1.30% at
  // Beginner" beside a table reading 0.45% would be plainly wrong, and
  // paraphrasing it into something vague enough to cover both would lose the
  // point of having it.
  const trendNote =
    (activePolicy ?? "area") !== "area"
      ? null
      : TREND_NOTES[kind] ??
        (levelsWithData.size <= 1
          ? "Only evaluated at one board size so far -- not enough data yet to say how results change with board size or density."
          : null);

  const toggle = canToggle ? (
    <div className="flex flex-col gap-2">
      <div
        className="inline-flex w-fit rounded-lg border border-border p-0.5"
        role="group"
        aria-label="Board distribution"
      >
        {(Object.keys(FIRST_CLICK_POLICY_LABELS) as FirstClickPolicy[]).map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => setPolicy(option)}
            aria-pressed={policy === option}
            className={cn(
              "rounded-md px-3 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60",
              policy === option ? "text-white" : "text-text-muted hover:text-text",
            )}
            style={policy === option ? { backgroundColor: accentColor } : undefined}
          >
            {FIRST_CLICK_POLICY_LABELS[option]}
          </button>
        ))}
      </div>
      <p className="text-xs text-text-muted">{FIRST_CLICK_POLICY_BLURBS[policy]}</p>
    </div>
  ) : null;

  const header = (
    <>
      <h4 className="text-xs font-medium tracking-wide text-text-muted uppercase">Results across board size &amp; mine density</h4>
      {toggle}
    </>
  );

  if (status === "loading") {
    return (
      <div className="flex flex-col gap-3">
        {header}
        <Skeleton className="h-56 w-full" />
        {isSlow && <ColdStartNotice />}
      </div>
    );
  }
  if (status === "error" && error) {
    return (
      <div className="flex flex-col gap-3">
        {header}
        <ApiErrorState error={error} onRetry={retry} title="Couldn't load board-size results" />
      </div>
    );
  }
  if (!visibleRows) return null;

  return (
    <div className="flex flex-col gap-3">
      {header}

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
            {visibleRows.map((row) => {
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

      {trendNote && <NarrativeText value={trendNote} className="text-sm text-text-muted" />}

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
