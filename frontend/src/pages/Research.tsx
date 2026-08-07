import { useParams } from "react-router-dom";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { LevelDensitySelector } from "@/components/board/LevelDensitySelector";
import { ResearchPipeline } from "@/components/research/ResearchPipeline";
import { getLeaderboard } from "@/api/metrics";
import { useApiQuery } from "@/hooks/useApiQuery";
import { useBoardLevel } from "@/hooks/useBoardLevel";

/**
 * The interactive research pipeline: Random -> CSP -> Q-Learning -> DQN ->
 * PPO, each a milestone that expands into its own real experiment story
 * (why it was tried, what was tested, the results, what it couldn't do, and
 * what changed next). This is the project's centerpiece -- what used to be
 * a separate "Experiments" browser is now folded into whichever milestone
 * it belongs to.
 *
 * The level/density selector only changes the *quantitative* numbers (each
 * milestone's win-rate badge, sourced from the same leaderboard call) --
 * the hand-authored narrative (why each algorithm was tried, what
 * limitation it hit) stays fixed, since that's the historical research
 * story behind this project, not a live number that changes per board.
 */
export function Research() {
  const { agentSlug } = useParams<{ agentSlug?: string }>();
  const { configs, level, density, setLevel, setDensity } = useBoardLevel();
  const { data, status, error, isSlow, retry } = useApiQuery(() => getLeaderboard(level, density), [level, density]);

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight text-heading sm:text-4xl">The research pipeline</h1>
        <p className="mt-2 max-w-2xl text-text-muted">
          We started with a coin flip, discovered its limits, and kept building until a learned policy beat it.
          Click any milestone to open its full experiment story — problem, experiments, results, and what changed
          next.
        </p>
      </div>

      {configs.length > 0 && (
        <LevelDensitySelector configs={configs} level={level} density={density} onLevelChange={setLevel} onDensityChange={setDensity} />
      )}

      {status === "loading" && (
        <div className="flex flex-col gap-4">
          <Skeleton className="h-40 w-full" />
          <Skeleton className="h-64 w-full" />
          {isSlow && <ColdStartNotice />}
        </div>
      )}
      {status === "error" && error && <ApiErrorState error={error} onRetry={retry} title="Couldn't load the research pipeline" />}
      {status === "success" && data && <ResearchPipeline leaderboard={data} initialAgent={agentSlug ?? null} />}
    </div>
  );
}
