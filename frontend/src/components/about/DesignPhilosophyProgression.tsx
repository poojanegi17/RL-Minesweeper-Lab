import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { PipelineNode } from "@/components/research/PipelineNode";
import { PipelineConnector } from "@/components/research/PipelineConnector";
import { useTheme } from "@/app/ThemeProvider";
import { agentKindFromName, slugifyAgentName } from "@/lib/agentAdapters";
import { AGENT_HEX } from "@/data/types";

/** A one-line *rationale* per algorithm -- deliberately shorter than
 * `ResearchPipeline`'s `STEPS.whyAttempted` (which explains what problem in
 * the previous step motivated this one, at chapter length). This is a
 * teaser for that page, not a second copy of it -- every node links
 * straight to its real chapter on `/research`. */
const PHILOSOPHY = [
  { agent: "Random", rationale: "Establish a baseline floor." },
  { agent: "CSP", rationale: "Understand how far deterministic reasoning can go." },
  { agent: "Q-Learning", rationale: "Introduce learning from experience." },
  { agent: "DQN", rationale: "Learn spatial patterns instead of relying on tables." },
  { agent: "PPO", rationale: "Explore stable policy optimization." },
] as const;

/**
 * "Why does this project have five different algorithms?" -- the same
 * milestone-row visual `ResearchPipeline`/Home's teaser already use
 * (`PipelineNode`/`PipelineConnector`, reused as-is), but non-expanding:
 * each node is a plain link into its real chapter on `/research`, so this
 * never drifts out of sync with the actual research narrative.
 */
export function DesignPhilosophyProgression() {
  const { theme } = useTheme();

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="flex flex-col items-stretch gap-2 sm:flex-row sm:flex-wrap sm:items-center sm:justify-center">
        {PHILOSOPHY.map((step, index) => {
          const kind = agentKindFromName(step.agent);
          const accentColor = AGENT_HEX[kind][theme];

          return (
            <div key={step.agent} className="flex flex-col items-center sm:contents">
              <PipelineNode
                title={step.agent}
                kind={kind}
                accentColor={accentColor}
                winRate={null}
                subtitle={step.rationale}
                to={`/research/${slugifyAgentName(step.agent)}`}
                index={index}
              />
              {index < PHILOSOPHY.length - 1 && (
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

      <Link to="/research" className="flex items-center gap-1 text-sm font-medium text-primary hover:underline">
        Explore the full research pipeline
        <ArrowRight className="h-3.5 w-3.5" />
      </Link>
    </div>
  );
}
