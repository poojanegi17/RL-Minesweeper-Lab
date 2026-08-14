import { Link } from "react-router-dom";
import { useTheme } from "@/app/ThemeProvider";
import { PipelineFlowCard } from "@/components/research/PipelineFlowCard";
import { PipelineConnector } from "@/components/research/PipelineConnector";
import { STEPS } from "@/components/research/ResearchPipeline";
import { agentKindFromName, slugifyAgentName } from "@/lib/agentAdapters";
import { AGENT_EXPLAINERS } from "@/lib/agentExplainers";
import { bestWinRateFor, fetchAllBoardConfigLeaderboards } from "@/lib/boardComparison";
import { useApiQuery } from "@/hooks/useApiQuery";
import { AGENT_HEX } from "@/data/types";
import type { LeaderboardEntry } from "@/types/metrics";

interface ResearchJourneyProps {
  /** Real, already-fetched default-board leaderboard data (`GET
   * /api/leaderboard`) -- used as the win-rate fallback while this
   * component's own best-across-configs fetch is still in flight. */
  leaderboard: LeaderboardEntry[];
}

/**
 * A clickable teaser of the full research pipeline (`STEPS` from
 * `@/components/research/ResearchPipeline`, reused rather than redefined) --
 * the same vertical `PipelineFlowCard` look the `/research` page itself uses
 * (name + strategy tagline), just in link mode instead of expand-in-place, so
 * this stays a teaser rather than a second copy of the full chamber story.
 * Every card links straight to `/research/{slug}`, and the row still ends
 * with the same "explore the full pipeline" link it always has.
 */
export function ResearchJourney({ leaderboard }: ResearchJourneyProps) {
  const { theme } = useTheme();
  // Best win rate seen across every board size/density, not just the default
  // beginner/standard board -- matches what `/research`'s own cards show
  // (see `ResearchPipeline`), so the two don't disagree on the same agent.
  const { data: boardSnapshots } = useApiQuery(fetchAllBoardConfigLeaderboards, []);

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="flex w-full flex-col">
        {STEPS.map((step, index) => {
          const kind = agentKindFromName(step.agent);
          const accentColor = AGENT_HEX[kind][theme];
          const defaultWinRate = leaderboard.find((entry) => entry.agent === step.agent)?.win_rate ?? null;
          const winRate = boardSnapshots ? (bestWinRateFor(step.agent, boardSnapshots) ?? defaultWinRate) : defaultWinRate;
          const strategy = AGENT_EXPLAINERS[kind].tagline;

          return (
            <div key={step.agent} className="flex flex-col">
              <PipelineFlowCard
                title={step.title}
                kind={kind}
                accentColor={accentColor}
                strategy={strategy}
                winRate={winRate}
                to={`/research/${slugifyAgentName(step.agent)}`}
                index={index}
              />
              {index < STEPS.length - 1 && (
                <div className="flex flex-col items-center py-1">
                  <PipelineConnector orientation="vertical" />
                </div>
              )}
            </div>
          );
        })}
      </div>

      <Link to="/research" className="text-sm font-medium text-primary hover:underline">
        Explore the full research pipeline →
      </Link>
    </div>
  );
}
