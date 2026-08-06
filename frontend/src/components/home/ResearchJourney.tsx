import { Link } from "react-router-dom";
import { useTheme } from "@/app/ThemeProvider";
import { PipelineNode } from "@/components/research/PipelineNode";
import { PipelineConnector } from "@/components/research/PipelineConnector";
import { STEPS } from "@/components/research/ResearchPipeline";
import { agentKindFromName, slugifyAgentName } from "@/lib/agentAdapters";
import { AGENT_HEX } from "@/data/types";
import type { LeaderboardEntry } from "@/types/metrics";

interface ResearchJourneyProps {
  /** Real, already-fetched leaderboard data (`GET /api/leaderboard`) -- each
   * milestone's badge is this agent's actual current win rate, never a
   * hardcoded or remembered number. */
  leaderboard: LeaderboardEntry[];
}

/**
 * A compact, clickable preview of the full research pipeline (`STEPS` from
 * `@/components/research/ResearchPipeline`, reused rather than redefined) --
 * every node links straight to `/research/{slug}`, where the full
 * interactive experiment story lives. Home no longer tries to tell the
 * whole story inline.
 */
export function ResearchJourney({ leaderboard }: ResearchJourneyProps) {
  const { theme } = useTheme();

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="flex flex-col items-stretch gap-2 sm:flex-row sm:flex-wrap sm:items-center sm:justify-center">
        {STEPS.map((step, index) => {
          const kind = agentKindFromName(step.agent);
          const accentColor = AGENT_HEX[kind][theme];
          const winRate = leaderboard.find((entry) => entry.agent === step.agent)?.win_rate ?? null;

          return (
            <div key={step.agent} className="flex flex-col items-center sm:contents">
              <PipelineNode
                title={step.title}
                kind={kind}
                accentColor={accentColor}
                winRate={winRate}
                to={`/research/${slugifyAgentName(step.agent)}`}
                index={index}
              />
              {index < STEPS.length - 1 && (
                <>
                  <div className="sm:hidden">
                    <PipelineConnector orientation="vertical" />
                  </div>
                  <div className="hidden sm:block">
                    <PipelineConnector orientation="horizontal" />
                  </div>
                </>
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
