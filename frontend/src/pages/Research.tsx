import { useParams } from "react-router-dom";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { ResearchPipeline } from "@/components/research/ResearchPipeline";
import { getLeaderboard } from "@/api/metrics";
import { useApiQuery } from "@/hooks/useApiQuery";

/**
 * The interactive research pipeline: Random -> CSP -> Q-Learning -> DQN ->
 * PPO, each a milestone that expands into its own real experiment story
 * (why it was tried, what was tested, the results, what it couldn't do, and
 * what changed next). This is the project's centerpiece -- what used to be
 * a separate "Experiments" browser is now folded into whichever milestone
 * it belongs to.
 *
 * The pipeline row itself always shows the default beginner/standard board
 * (no page-wide level toggle) -- each milestone's own expanded chamber has
 * its own independent level/density selector (see `ExperimentChamber`),
 * since "what board is CSP's chamber showing" and "what board is DQN's
 * chamber showing" are genuinely independent questions.
 */
export function Research() {
  const { agentSlug } = useParams<{ agentSlug?: string }>();
  const { data, status, error, isSlow, retry } = useApiQuery(getLeaderboard, []);

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
