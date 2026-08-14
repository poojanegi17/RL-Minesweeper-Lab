import { useState } from "react";
import { motion } from "framer-motion";
import { Trophy } from "lucide-react";
import { SectionEyebrow } from "@/components/landing/LandingPrimitives";
import { AGENT_ICONS } from "@/components/agent/agentIcons";
import { Skeleton } from "@/components/ui/Skeleton";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { LevelDensitySelector, DENSITY_LABELS } from "@/components/board/LevelDensitySelector";
import { getLeaderboard } from "@/api/metrics";
import { useApiQuery } from "@/hooks/useApiQuery";
import { useBoardLevel } from "@/hooks/useBoardLevel";
import { agentKindFromName } from "@/lib/agentAdapters";
import { formatPercent } from "@/lib/experimentAdapters";
import {
  bestRunForEnv,
  fetchLevelPipeline,
  LEVEL_LABELS_FULL,
  LEVEL_PIPELINE_IDS,
  pipelineIdForEnv,
  POLICY_ENV_VERSION,
  visibleStories,
  type PipelineLevel,
} from "@/lib/levelPipelines";
import { FIRST_CLICK_POLICY_LABELS, type FirstClickPolicy } from "@/lib/boardLevelQuery";
import { AGENT_HEX } from "@/data/types";
import { cn } from "@/lib/cn";

const POLICIES: FirstClickPolicy[] = ["none", "area"];
/** Fixed axis ticks -- see the scale note in the component docstring. */
const TICKS = [0, 25, 50, 75, 100];

/**
 * "Who leads here?" -- the leaderboard as a bar chart that re-reads itself
 * whenever the board configuration changes, plus the name of the model
 * producing the winning number.
 *
 * The axis is pinned to 0-100% rather than scaled to the tallest bar. Scaling
 * to the max would make every configuration look similar -- the leader's bar
 * always full width -- which hides the thing this section exists to show: CSP
 * clears 98.55% of sparse 9x9 boards and 6.30% of dense 16x16 ones, and that
 * collapse is the point. A fixed axis also keeps the comparison honest when
 * the visitor flips between environments, where the same agent moves by tens
 * of points. Values are printed at the end of each bar so the near-zero rows
 * stay readable.
 *
 * The winning model is resolved through the research pipeline, the same path
 * the agent, compare and replay pages use, so all four name the same run for a
 * given configuration rather than each deriving it their own way.
 */
export function WhoLeadsSection() {
  const { configs, level, density, setLevel, setDensity } = useBoardLevel();
  const [policy, setPolicy] = useState<FirstClickPolicy>("none");
  const pipelineLevel = level as PipelineLevel;
  const env = POLICY_ENV_VERSION[policy];

  const { data, status, error, retry } = useApiQuery(async () => {
    const leaderboard = await getLeaderboard(level, density, policy);
    const ranked = [...leaderboard].sort((a, b) => (b.win_rate ?? -1) - (a.win_rate ?? -1));
    const leader = ranked.find((entry) => entry.win_rate != null) ?? null;

    // Only the learned agents have a run to name; CSP/Random/Q-Learning
    // deliberately resolve to null rather than being given a fabricated one.
    const models: Record<string, string> = {};
    await Promise.all(
      ["DQN", "PPO"].map(async (agent) => {
        const levelId = LEVEL_PIPELINE_IDS[agent]?.[pipelineLevel];
        if (!levelId) return;
        const envPipeline = pipelineIdForEnv(agent, pipelineLevel, env);
        const pipeline = await fetchLevelPipeline(envPipeline?.id ?? levelId);
        const shown = envPipeline ? pipeline.stories : visibleStories(pipeline.stories, agent, pipelineLevel);
        const picked = bestRunForEnv(shown, env);
        if (picked) models[agent] = picked.story.runBrief.id;
      }),
    );

    return { ranked, leader, models };
  }, [level, density, policy, pipelineLevel, env]);

  return (
    <section
      id="who-leads"
      className="relative z-10 mx-auto flex min-h-screen max-w-6xl flex-col justify-center px-6 py-16 md:py-24"
    >
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        className="liquid-glass relative overflow-hidden rounded-3xl border border-white/10 shadow-[0_0_120px_-20px_rgba(0,210,255,0.25)]"
      >
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-5 py-4">
          <SectionEyebrow label="Who leads here?" tag="changes with the board" />
          <span className="inline-flex items-center gap-1.5 rounded-full border border-white/10 px-2.5 py-1 text-[10px] tracking-wide text-white/50 uppercase">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" aria-hidden="true" />
            Live data
          </span>
        </div>

        <div className="glossy-scope flex flex-col gap-5 p-5">
          {/* Environment first, then board -- the same order of consequence the
           * rest of the site uses: this control changes the game, the ones
           * below change the configuration within it. */}
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="inline-flex rounded-xl border border-white/10 p-1" role="group" aria-label="Environment">
              {POLICIES.map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setPolicy(option)}
                  aria-pressed={policy === option}
                  className={cn(
                    "rounded-lg px-4 py-1.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60",
                    policy === option ? "bg-primary text-white shadow-sm" : "text-text-muted hover:text-text",
                  )}
                >
                  {FIRST_CLICK_POLICY_LABELS[option]}
                </button>
              ))}
            </div>
            <p className="text-xs text-text-muted">
              Two different games — a win rate under one is not comparable to one under the other.
            </p>
          </div>

          {configs.length > 0 && (
            <LevelDensitySelector
              configs={configs}
              level={level}
              density={density}
              onLevelChange={setLevel}
              onDensityChange={setDensity}
            />
          )}

          {status === "loading" && <Skeleton className="h-72 w-full" />}
          {status === "error" && error && (
            <ApiErrorState error={error} onRetry={retry} title="Couldn't load this configuration" />
          )}

          {status === "success" && data && (
            <>
              <LeaderCallout
                agent={data.leader?.agent ?? null}
                winRate={data.leader?.win_rate ?? null}
                modelId={data.leader ? data.models[data.leader.agent] : undefined}
                cell={`${LEVEL_LABELS_FULL[pipelineLevel] ?? level} · ${DENSITY_LABELS[density] ?? density} · ${FIRST_CLICK_POLICY_LABELS[policy].toLowerCase()}`}
              />

              <WinRateBars rows={data.ranked} models={data.models} />

              <p className="text-xs text-text-muted">
                2,000 evaluation episodes per agent, seed 42 — every agent faces the identical boards at this cell.
                Agents with no run at this board size are shown as not trained rather than as zero.
              </p>
            </>
          )}
        </div>
      </motion.div>
    </section>
  );
}

function LeaderCallout({
  agent,
  winRate,
  modelId,
  cell,
}: {
  agent: string | null;
  winRate: number | null;
  modelId?: string;
  cell: string;
}) {
  if (!agent) {
    return <p className="text-sm text-text-muted">No agent has a measured result at {cell} yet.</p>;
  }
  const kind = agentKindFromName(agent);
  const Icon = AGENT_ICONS[kind];
  const color = AGENT_HEX[kind].dark;

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-xl border border-white/10 px-4 py-3">
      <span
        className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg"
        style={{ backgroundColor: `${color}1a`, color }}
      >
        <Icon className="h-5 w-5" />
      </span>
      <div className="min-w-0 flex-1">
        <p className="flex flex-wrap items-center gap-2 font-semibold text-heading">
          <Trophy className="h-4 w-4 shrink-0" style={{ color }} aria-hidden="true" />
          {agent} leads at {cell}
        </p>
        <p className="mt-0.5 truncate text-xs text-text-muted">
          {modelId ? (
            <>
              Best model here: <span className="font-mono text-text">{modelId}</span>
            </>
          ) : (
            "No trained model — this agent solves the board directly rather than learning it."
          )}
        </p>
      </div>
      <span className="font-mono text-2xl font-semibold tabular-nums" style={{ color }}>
        {formatPercent(winRate)}
      </span>
    </div>
  );
}

function WinRateBars({
  rows,
  models,
}: {
  rows: { agent: string; win_rate: number | null; source: string }[];
  models: Record<string, string>;
}) {
  const best = rows.reduce<number>((acc, row) => Math.max(acc, row.win_rate ?? -1), -1);

  return (
    <div className="flex flex-col gap-2">
      {/* One plot area, not one box per bar. The gridline layer is a single grid
       * item spanning every row of the bar column, so the ticks run continuously
       * behind all five bars -- that is what makes this read as a chart rather
       * than five separate meters stacked up. Bars are declared after it in DOM
       * order, so they paint on top without needing a z-index. */}
      <div
        className="relative grid items-center gap-x-3 gap-y-1"
        style={{ gridTemplateColumns: "5.5rem minmax(0,1fr) 4rem" }}
      >
        <div
          className="pointer-events-none relative h-full"
          style={{ gridColumn: 2, gridRow: `1 / ${rows.length + 1}` }}
          aria-hidden="true"
        >
          {TICKS.map((tick) => (
            <span
              key={tick}
              className={cn("absolute inset-y-0 w-px", tick === 0 ? "bg-white/20" : "bg-white/[0.07]")}
              style={{ left: `${tick}%` }}
            />
          ))}
        </div>

        {rows.map((row, index) => {
          const kind = agentKindFromName(row.agent);
          const color = AGENT_HEX[kind].dark;
          const trained = row.win_rate != null;
          // A 0.6% floor so a non-zero result never renders as an invisible
          // sliver; the printed value beside it remains the exact figure.
          const width = trained ? Math.max(row.win_rate! * 100, 0.6) : 0;
          const isLeader = trained && row.win_rate === best;

          return (
            <div key={row.agent} className="group contents">
              <span
                className="truncate text-right text-sm text-text-muted transition-colors group-hover:text-text"
                style={{ gridColumn: 1, gridRow: index + 1 }}
                title={row.agent}
              >
                {row.agent}
              </span>

              <div className="relative h-8 min-w-0 py-1" style={{ gridColumn: 2, gridRow: index + 1 }}>
                {trained ? (
                  <motion.div
                    className="h-full rounded-r-[4px]"
                    style={{
                      backgroundColor: color,
                      boxShadow: isLeader ? `0 0 20px -4px ${color}` : undefined,
                    }}
                    initial={{ width: 0 }}
                    animate={{ width: `${width}%` }}
                    transition={{ duration: 0.6, delay: index * 0.06, ease: "easeOut" }}
                  />
                ) : (
                  <span className="flex h-full items-center pl-2 text-xs text-text-muted italic">
                    not trained at this board size
                  </span>
                )}

                {/* Hover detail. Anchored to the row rather than the cursor so it
                 * never covers the bar it describes. */}
                <div className="pointer-events-none absolute -top-1 left-0 z-20 hidden -translate-y-full group-hover:block">
                  <div className="rounded-lg border border-white/15 bg-black/90 px-3 py-2 whitespace-nowrap shadow-xl">
                    <p className="flex items-center gap-2 text-xs font-medium text-white">
                      <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} aria-hidden="true" />
                      {row.agent}
                      <span className="font-mono tabular-nums">{trained ? formatPercent(row.win_rate) : "not trained"}</span>
                    </p>
                    <p className="mt-0.5 font-mono text-[10px] text-white/50">
                      {models[row.agent] ?? "no trained model — solves the board directly"}
                    </p>
                  </div>
                </div>
              </div>

              <span
                className="text-right font-mono text-sm tabular-nums"
                style={{ gridColumn: 3, gridRow: index + 1, color: trained ? color : undefined }}
              >
                {trained ? formatPercent(row.win_rate) : "—"}
              </span>
            </div>
          );
        })}
      </div>

      {/* Axis. Pinned 0-100 rather than scaled to the leader -- see docstring. */}
      <div className="grid items-center gap-x-3" style={{ gridTemplateColumns: "5.5rem minmax(0,1fr) 4rem" }}>
        <span aria-hidden="true" />
        <div className="relative h-4">
          {TICKS.map((tick) => (
            <span
              key={tick}
              className="absolute top-0 font-mono text-[10px] text-text-muted"
              style={{ left: `${tick}%`, transform: tick === 0 ? undefined : "translateX(-50%)" }}
            >
              {tick}%
            </span>
          ))}
        </div>
        <span aria-hidden="true" />
      </div>

      {/* Also shown on hover, but kept here too: hover does not exist on touch,
       * and which checkpoint produced a number is provenance this project does
       * not hide behind an interaction. */}
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1">
        {rows
          .filter((row) => models[row.agent])
          .map((row) => (
            <span key={row.agent} className="font-mono text-[11px] text-text-muted">
              {row.agent}: {models[row.agent]}
            </span>
          ))}
      </div>
    </div>
  );
}
