import { Link, useParams } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { LevelStory } from "@/components/research/LevelStory";
import { NotFound } from "@/pages/NotFound";
import { useTheme } from "@/app/ThemeProvider";
import { agentKindFromName, slugifyAgentName } from "@/lib/agentAdapters";
import { LEVEL_LABELS_FULL, LEVEL_PIPELINE_IDS, type PipelineLevel } from "@/lib/levelPipelines";
import { AGENT_HEX } from "@/data/types";

const AGENT_NAMES = ["Random", "CSP", "Q-Learning", "DQN", "PPO"];
const LEVELS: PipelineLevel[] = ["beginner", "intermediate", "expert"];

/** Board dimensions per level, for the page subtitle -- mirrors
 * `rl/board_configs.py`'s catalog, same duplication the backend already
 * accepts (see `app/board_levels.py`). */
const LEVEL_BOARDS: Record<PipelineLevel, string> = {
  beginner: "5×5",
  intermediate: "9×9",
  expert: "16×16",
};

/**
 * One agent's research story at one board size, at page scale --
 * `/research/:agentSlug/:level`.
 *
 * Split out of the research chamber because these stories outgrew a
 * collapsible card: DQN Beginner alone is five chapters plus ten runs with
 * their own training curves and hyperparameter diffs. A dedicated route also
 * makes a level linkable, which a nested accordion never was.
 *
 * Renders nothing of its own beyond the header -- `LevelStory` decides
 * whether this level is told as chapters or as one card per run.
 */
export function LevelDetail() {
  const { agentSlug, level } = useParams<{ agentSlug: string; level: string }>();
  const { theme } = useTheme();

  const agentName = AGENT_NAMES.find((name) => slugifyAgentName(name) === agentSlug);
  const pipelineLevel = LEVELS.find((candidate) => candidate === level);
  const levelId = agentName && pipelineLevel ? LEVEL_PIPELINE_IDS[agentName]?.[pipelineLevel] : undefined;

  // An agent/level pair with no pipeline id is one nobody has trained at that
  // size yet -- a real 404 rather than an empty page, matching how
  // `ExperimentSetup` simply omits the link.
  if (!agentName || !pipelineLevel || !levelId) return <NotFound />;

  const accentColor = AGENT_HEX[agentKindFromName(agentName)][theme];

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col gap-3">
        <Link
          to={`/research/${slugifyAgentName(agentName)}`}
          className="inline-flex w-fit items-center gap-1.5 text-sm text-text-muted transition-colors hover:text-text"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to {agentName}&apos;s research
        </Link>

        <div>
          <p className="text-xs font-medium tracking-wide uppercase" style={{ color: accentColor }}>
            {agentName} · {LEVEL_BOARDS[pipelineLevel]}
          </p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight text-heading sm:text-4xl">
            {LEVEL_LABELS_FULL[pipelineLevel]}
          </h1>
          <p className="mt-2 max-w-2xl text-text-muted">
            How this agent was actually built at this board size — every configuration tried, what each one showed,
            and what was still wrong with it.
          </p>
        </div>
      </div>

      <LevelStory level={pipelineLevel} levelId={levelId} agentName={agentName} accentColor={accentColor} />
    </div>
  );
}
