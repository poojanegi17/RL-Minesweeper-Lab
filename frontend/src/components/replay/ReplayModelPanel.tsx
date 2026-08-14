import { AlertTriangle, Trophy } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { DENSITY_LABELS } from "@/components/board/LevelDensitySelector";
import { getLeaderboard } from "@/api/metrics";
import { useApiQuery } from "@/hooks/useApiQuery";
import { formatPercent, humanizeVariant, type VariantStory } from "@/lib/experimentAdapters";
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

/** Agents that train nothing, with what they do instead -- so the panel says
 * why there is no model rather than leaving an empty section. */
const UNTRAINED_BEHAVIOUR: Record<string, string> = {
  CSP: "Solves constraints directly from the revealed numbers. No training, no weights, and no episode budget — it plays the same way on its first board as its millionth.",
  Random: "Picks uniformly among hidden cells. Included as the floor every other agent has to clear.",
  "Q-Learning": "Learns a lookup table keyed by exact board pattern. It trains, but persists no checkpoint anywhere in this project, so there is nothing to replay.",
};

interface ReplayModelPanelProps {
  agentName: string;
  level: string;
  density: string;
  policy: FirstClickPolicy;
  /** The `experiment_id` recorded on the episode currently being watched, so
   * the panel can verify the episode really came from the model it describes
   * rather than asserting it. */
  replayExperimentId: string | null;
  accentColor: string;
}

/**
 * Which model produced the episode on screen, and how well it does at this
 * exact board configuration.
 *
 * The replay board shows one episode, which is a sample of size one -- a win
 * looks like competence and a loss looks like failure regardless of how the
 * agent actually performs. Pairing every episode with the model's measured win
 * rate at the same (level, density, environment) is what stops a single lucky
 * or unlucky board from reading as the result.
 *
 * The model is resolved through the research pipeline (`LEVEL_PIPELINE_IDS` +
 * `bestRunForEnv`), the same path the agent page's best-model card and the
 * compare page use, so all three name the same run for a given configuration.
 * The win rate comes from the leaderboard scoped to that cell, which is the
 * same figure the compare page's stat row shows.
 */
export function ReplayModelPanel({
  agentName,
  level,
  density,
  policy,
  replayExperimentId,
  accentColor,
}: ReplayModelPanelProps) {
  const pipelineLevel = level as PipelineLevel;
  const env = POLICY_ENV_VERSION[policy];

  const { data, status } = useApiQuery(async () => {
    const leaderboard = await getLeaderboard(level, density, policy);
    const winRate = leaderboard.find((entry) => entry.agent === agentName)?.win_rate ?? null;

    const levelId = LEVEL_PIPELINE_IDS[agentName]?.[pipelineLevel];
    if (!levelId) return { winRate, story: null as VariantStory | null, isTransfer: false };

    const envPipeline = pipelineIdForEnv(agentName, pipelineLevel, env);
    const pipeline = await fetchLevelPipeline(envPipeline?.id ?? levelId);
    const shown = envPipeline ? pipeline.stories : visibleStories(pipeline.stories, agentName, pipelineLevel);
    const picked = bestRunForEnv(shown, env);
    return { winRate, story: picked?.story ?? null, isTransfer: picked?.isTransfer ?? false };
  }, [agentName, level, density, policy, pipelineLevel, env]);

  if (status === "loading") return <Skeleton className="h-24 w-full" />;
  if (!data) return null;

  const { winRate, story, isTransfer } = data;
  const cell = `${LEVEL_LABELS_FULL[pipelineLevel] ?? level} · ${DENSITY_LABELS[density] ?? density} · ${FIRST_CLICK_POLICY_LABELS[policy].toLowerCase()}`;
  const untrained = UNTRAINED_BEHAVIOUR[agentName];

  // The episode is only evidence about this model if it was actually produced
  // by it. Checked rather than assumed -- a regenerated board-result tree and a
  // stale replay directory would otherwise disagree silently.
  const mismatched = story != null && replayExperimentId != null && replayExperimentId !== story.runBrief.id;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-xs font-medium tracking-wide text-text-muted uppercase">
            {story ? "Best found model at this configuration" : "Agent"}
          </h3>
          <p className="mt-1 truncate font-semibold text-heading">
            {story
              ? story.runBrief.variant
                ? humanizeVariant(story.runBrief.variant)
                : story.runBrief.title
              : agentName}
          </p>
          {story && <p className="mt-0.5 truncate font-mono text-xs text-text-muted">{story.runBrief.id}</p>}
        </div>

        <div className="shrink-0 text-right">
          <p className="font-mono text-2xl font-semibold tabular-nums" style={{ color: accentColor }}>
            {formatPercent(winRate)}
          </p>
          <p className="text-[11px] text-text-muted">win rate here</p>
        </div>
      </div>

      <p className="text-xs text-text-muted">{cell}</p>

      {story ? (
        <p className="text-sm text-text-muted">
          Trained for {story.detail.episodes.toLocaleString()} episodes on a {story.detail.board} board (
          {story.detail.mines} mines)
          {isTransfer
            ? ", under the other opening rule — shown here as a generalization result"
            : ""}
          . The win rate is 2,000 greedy episodes at seed 42 on this exact configuration, so the single episode on the
          board is one draw from it rather than a summary of it.
        </p>
      ) : (
        <p className="text-sm text-text-muted">{untrained ?? `${agentName} has no trained model at this board size.`}</p>
      )}

      {story && (
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge variant="neutral" className="gap-1 text-primary">
            <Trophy className="h-3 w-3" />
            Best at this config
          </Badge>
          {isTransfer && <Badge variant="outline">Generalization result</Badge>}
        </div>
      )}

      {mismatched && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-xs text-text-muted">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" aria-hidden="true" />
          <p>
            This episode was recorded from <span className="font-mono text-text">{replayExperimentId}</span>, not the
            model named above. Regenerate this configuration's replays so the episode and the win rate describe the
            same weights.
          </p>
        </div>
      )}
    </div>
  );
}
