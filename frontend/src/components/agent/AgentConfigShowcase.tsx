import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Trophy, XCircle } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { ReplayBoard } from "@/components/replay/ReplayBoard";
import { ReplayControls } from "@/components/replay/ReplayControls";
import { LevelDensitySelector } from "@/components/board/LevelDensitySelector";
import { EnvironmentToggle } from "@/components/compare/EnvironmentToggle";
import { getBoardConfigs } from "@/api/boardConfigs";
import { getLeaderboard } from "@/api/metrics";
import { getReplay, getReplays } from "@/api/replays";
import { useApiQuery } from "@/hooks/useApiQuery";
import { formatPercent, formatReward } from "@/lib/experimentAdapters";
import { type FirstClickPolicy } from "@/lib/boardLevelQuery";
import type { ReplayDetail, ReplaySummary } from "@/types/replay";

interface AgentConfigShowcaseProps {
  agentName: string;
  accentColor: string;
  /** Controlled from `AgentDetail` (not local state here) so the "Best found
   * model at this config" card above can react to the exact same selection
   * this section's own results/replay already do -- one selection, not two
   * independently-toggled copies of it. */
  policy: FirstClickPolicy;
  level: string;
  density: string;
  onPolicyChange: (policy: FirstClickPolicy) => void;
  onLevelChange: (level: string) => void;
  onDensityChange: (density: string) => void;
}

/**
 * Picks the single episode worth showing for a board cell.
 *
 * The longest win, because a win is what a visitor came to see and the
 * longest one shows the most decision-making rather than a one-click cascade.
 * Where the agent never wins at this configuration -- Random genuinely does
 * not, across its recorded dense-density episodes -- the longest *loss* is
 * shown instead, clearly labelled. Substituting a win from an easier cell
 * would misrepresent the result, and showing nothing would hide it.
 */
function pickFeatured(replays: ReplaySummary[]): { replay: ReplaySummary; isFallback: boolean } | null {
  if (replays.length === 0) return null;
  const byLength = [...replays].sort((a, b) => b.steps - a.steps);
  const win = byLength.find((r) => r.won);
  return win ? { replay: win, isFallback: false } : { replay: byLength[0], isFallback: true };
}

/**
 * One agent, shown at whichever board configuration the visitor selects:
 * its headline numbers there, and a single recorded episode of it playing.
 *
 * The environment sits above board size and density for the same reason it
 * does on the Compare page -- `first_click_safe` changes the game rather than
 * its configuration, and an episode recorded under one distribution must
 * never be presented as play from the other.
 */
export function AgentConfigShowcase({
  agentName,
  accentColor,
  policy,
  level,
  density,
  onPolicyChange,
  onLevelChange,
  onDensityChange,
}: AgentConfigShowcaseProps) {
  const { data, status, error, isSlow, retry } = useApiQuery(async () => {
    const [configs, leaderboard, replays] = await Promise.all([
      getBoardConfigs(),
      getLeaderboard(level, density, policy),
      getReplays(level, density, policy),
    ]);
    const mine = replays.filter((r) => r.agent === agentName);
    const featured = pickFeatured(mine);
    const detail: ReplayDetail | null = featured
      ? await getReplay(featured.replay.id, level, density, policy)
      : null;
    return {
      configs,
      entry: leaderboard.find((e) => e.agent === agentName),
      total: mine.length,
      featured,
      detail,
    };
  }, [agentName, level, density, policy]);

  return (
    <div className="flex flex-col gap-5">
      <EnvironmentToggle value={policy} onChange={onPolicyChange} />

      {status === "loading" && (
        <div className="flex flex-col gap-3">
          <Skeleton className="h-64 w-full" />
          {isSlow && <ColdStartNotice />}
        </div>
      )}
      {status === "error" && error && <ApiErrorState error={error} onRetry={retry} title="Couldn't load this configuration" />}

      {status === "success" && data && (
        <>
          <LevelDensitySelector
            configs={data.configs}
            level={level}
            density={density}
            onLevelChange={onLevelChange}
            onDensityChange={onDensityChange}
          />

          <Card>
            <h3 className="text-xs font-medium tracking-wide text-text-muted uppercase">Results at this configuration</h3>
            {data.entry?.win_rate != null ? (
              <dl className="mt-3 flex flex-wrap gap-8 text-sm">
                <div>
                  <dt className="text-xs text-text-muted">Win rate</dt>
                  <dd className="font-mono text-lg font-semibold" style={{ color: accentColor }}>
                    {formatPercent(data.entry.win_rate)}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs text-text-muted">Avg. episode length</dt>
                  <dd className="font-mono font-medium text-text">{data.entry.avg_episode_length?.toFixed(2) ?? "—"}</dd>
                </div>
                <div>
                  <dt className="text-xs text-text-muted">Avg. reward</dt>
                  <dd className="font-mono font-medium text-text">{formatReward(data.entry.avg_reward)}</dd>
                </div>
              </dl>
            ) : (
              <p className="mt-2 text-sm text-text-muted">
                {agentName} has no recorded result at this board size and density.
              </p>
            )}
            <p className="mt-3 text-xs text-text-muted">2,000 evaluation episodes, seed 42.</p>
          </Card>

          <FeaturedReplay
            data={data}
            agentName={agentName}
            accentColor={accentColor}
            level={level}
            density={density}
          />
        </>
      )}
    </div>
  );
}

function FeaturedReplay({
  data,
  agentName,
  accentColor,
  level,
  density,
}: {
  data: { total: number; featured: { replay: ReplaySummary; isFallback: boolean } | null; detail: ReplayDetail | null };
  agentName: string;
  accentColor: string;
  level: string;
  density: string;
}) {
  const [stepIndex, setStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  const steps = useMemo(() => data.detail?.timeline ?? [], [data.detail]);

  useEffect(() => {
    setStepIndex(0);
    setIsPlaying(false);
  }, [data.detail]);

  useEffect(() => {
    if (!isPlaying || steps.length === 0) return;
    if (stepIndex >= steps.length - 1) {
      setIsPlaying(false);
      return;
    }
    const timer = setTimeout(() => setStepIndex((i) => i + 1), 700 / speed);
    return () => clearTimeout(timer);
  }, [isPlaying, stepIndex, steps.length, speed]);

  if (!data.featured || !data.detail) {
    return (
      <Card>
        <h3 className="text-xs font-medium tracking-wide text-text-muted uppercase">See it in action</h3>
        <p className="mt-2 text-sm text-text-muted">
          No episodes have been recorded for {agentName} at this configuration yet.
        </p>
      </Card>
    );
  }

  const step = steps[stepIndex];
  const { isFallback } = data.featured;

  return (
    <Card className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-xs font-medium tracking-wide text-text-muted uppercase">See it in action</h3>
        {isFallback ? (
          <Badge variant="outline" className="gap-1">
            <XCircle className="h-3 w-3" />
            Longest loss — no win recorded here
          </Badge>
        ) : (
          <Badge variant="neutral" className="gap-1" style={{ color: accentColor }}>
            <Trophy className="h-3 w-3" />
            Longest win
          </Badge>
        )}
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <div className="flex flex-col gap-3">
          <ReplayBoard
            board={step?.board_state ?? data.detail.initial_board}
            highlightedCell={step ? { row: step.action.row, col: step.action.col } : undefined}
            mineHit={Boolean(step?.done) && !data.detail.won}
          />
          <ReplayControls
            stepIndex={stepIndex}
            totalSteps={steps.length}
            isPlaying={isPlaying}
            speed={speed}
            onPrevious={() => setStepIndex((i) => Math.max(0, i - 1))}
            onNext={() => setStepIndex((i) => Math.min(steps.length - 1, i + 1))}
            onTogglePlay={() => setIsPlaying((p) => !p)}
            onSpeedChange={setSpeed}
          />
        </div>

        <div className="flex flex-col gap-3">
          <div>
            <p className="text-xs font-medium tracking-wide text-text-muted uppercase">Its reasoning</p>
            {step?.reasoning ? (
              <dl className="mt-2 flex flex-col gap-1.5 text-sm">
                {Object.entries(step.reasoning)
                  // `constraint_cells`/`remaining_mines` (CSP-only fields) are
                  // dropped from this at-a-glance dump -- often null and, even
                  // when present, too raw to read at a glance here; `inference`
                  // already states the same conclusion in a sentence.
                  .filter(([key]) => key !== "constraint_cells" && key !== "remaining_mines")
                  .map(([key, value]) => (
                    <div key={key} className="flex justify-between gap-3">
                      <dt className="text-text-muted">{key.replace(/_/g, " ")}</dt>
                      <dd className="text-right font-mono text-text">{String(value)}</dd>
                    </div>
                  ))}
              </dl>
            ) : agentName === "Random" ? (
              // Only Random gets this specific explanation -- "no reasoning
              // recorded" is true for any agent, but *why* (uniform random
              // selection) is a fact about Random specifically, not a safe
              // default for whichever agent happens to have no reasoning on
              // a given step (e.g. Q-Learning, if it ever gets replays).
              <p className="mt-2 text-sm text-text-muted">
                {agentName} records no reasoning — it picks uniformly at random among hidden cells, so there is nothing
                to explain per move.
              </p>
            ) : (
              <p className="mt-2 text-sm text-text-muted">No reasoning recorded for this step.</p>
            )}
          </div>

          <div className="rounded-lg border border-border px-3 py-2 text-xs text-text-muted">
            Episode {data.detail.episode_number} · {data.detail.steps} moves ·{" "}
            {data.detail.won ? "won" : "lost"} · reward {formatReward(data.detail.total_reward)}
          </div>
        </div>
      </div>

      <Link
        to={`/replay?level=${level}&density=${density}`}
        className="inline-flex w-fit items-center gap-1.5 text-sm font-medium transition-colors hover:underline"
        style={{ color: accentColor }}
      >
        Watch more episodes at this configuration ({data.total} recorded)
        <ArrowRight className="h-3.5 w-3.5" />
      </Link>
    </Card>
  );
}
