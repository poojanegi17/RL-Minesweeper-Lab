import { useState } from "react";
import { ChevronDown, Info, Trophy } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { VariantStoryCard } from "@/components/research/VariantStoryCard";
import {
  bestRunForEnv,
  fetchLevelPipeline,
  pipelineIdForEnv,
  POLICY_ENV_VERSION,
  visibleStories,
  type PipelineLevel,
} from "@/lib/levelPipelines";
import { formatPercent, humanizeVariant } from "@/lib/experimentAdapters";
import { FIRST_CLICK_POLICY_LABELS, type FirstClickPolicy } from "@/lib/boardLevelQuery";
import type { EnvVersion } from "@/types/experiment";
import { useApiQuery } from "@/hooks/useApiQuery";
import { cn } from "@/lib/cn";

/** How to name the distribution a transfer-case model was actually trained
 * under -- i.e. the *other* one from the currently selected `env`. */
const OTHER_ENV_LABEL: Record<EnvVersion, string> = {
  v2: "first click unsafe",
  v1: "first click safe",
};

interface BestModelCardProps {
  agentName: string;
  level: PipelineLevel;
  levelId: string;
  /** The first-click policy from the same board-config selection
   * `AgentConfigShowcase` is scoped to. Unlike density, this genuinely
   * selects a *different trained model* -- v1 and v2 are separate runs. */
  policy: FirstClickPolicy;
  accentColor: string;
}

/**
 * "Best found model at this config" -- the exact same collapsible detail
 * card `/research/{agent}/{level}` shows for its best-performing run
 * (`VariantStoryCard`: why it was tested, hyperparameters, training setup,
 * real training curves, its result across all three mine densities, and what
 * it taught us), not a reimplementation of it. Resolves the same way
 * `LevelStory` does (`fetchLevelPipeline` + `bestRunId`, falling back to the
 * highest win rate among visible stories if the API's pick got hidden), just
 * scoped to a single card instead of the whole level's chapter/variant list.
 *
 * **Scoped to (level, env), deliberately not to density.** Training happens at
 * Standard only; Sparse and Dense are the same finished network evaluated on
 * mine counts it never trained on. So there is exactly one best model per
 * (level, env), and the headline here is its own training-density result --
 * the number its confidence interval, curves, and hyperparameters all describe.
 * Following the density selector instead would have swapped in a figure from a
 * board the model was never fitted to while every other field on the card kept
 * describing Standard. The full density breakdown is not lost: it is
 * `VariantDensityTable` inside the expanded card, where it is labelled as the
 * generalization test it is.
 */
export function BestModelCard({ agentName, level, levelId, policy, accentColor }: BestModelCardProps) {
  const [open, setOpen] = useState(false);
  // The distribution the visitor is looking at, and the family that actually
  // trained under it -- which for Beginner is not the level's default family
  // (see `LEVEL_PIPELINE_ENV_IDS`).
  const env = POLICY_ENV_VERSION[policy];
  const envPipeline = pipelineIdForEnv(agentName, level, env);
  const resolvedId = envPipeline?.id ?? levelId;

  const { data: pipeline, status, error, isSlow, retry } = useApiQuery(
    () => fetchLevelPipeline(resolvedId),
    [resolvedId],
  );

  if (status === "loading") {
    return (
      <div className="flex flex-col gap-2">
        <Skeleton className="h-16 w-full" />
        {isSlow && <p className="text-xs text-text-muted">Waking up the backend…</p>}
      </div>
    );
  }
  if (status === "error" && error) {
    return <ApiErrorState error={error} onRetry={retry} title="Couldn't load this level's best model" />;
  }
  if (!pipeline) return null;

  // `HIDDEN_VARIANTS`/`VARIANT_DISPLAY_ORDER` describe the level's own v1
  // ablation family and are keyed by its variant names, so they must not be
  // applied to a dedicated env family: `dqn_v2_A_baseline`'s synthesized
  // variant is "baseline", which DQN Beginner hides -- filtering it would
  // remove the only run there is and take the card with it.
  const shown = envPipeline ? pipeline.stories : visibleStories(pipeline.stories, agentName, level);
  // Scoped to the selected distribution, not the family-wide max -- the win
  // rate beside this card and the episode above it both come from the tree
  // `policy` selects, so naming a run trained under the *other* distribution
  // would caption them with a model that never produced them.
  const picked = bestRunForEnv(shown, env);

  if (!picked) return null;
  const { story: best, isTransfer } = picked;

  const about = best.isBaseline
    ? pipeline.description
    : isTransfer
      ? `The best-performing configuration found for ${agentName} at this board size — trained under ${OTHER_ENV_LABEL[env]}, since no run exists under this one yet.`
      : `The best-performing configuration found for ${agentName} at this board size.`;

  return (
    <div className="flex flex-col gap-3">
      {/* Says up front why this one number ignores the density selector above
       * it -- otherwise a visitor switching to Dense and seeing it hold still
       * reads a bug rather than the methodology. */}
      <div className="flex items-start gap-2 rounded-lg border border-border px-3 py-2 text-xs text-text-muted">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        <p>
          One model is trained per board size and opening rule, always at Standard density — the other
          densities are evaluated afterwards on that same finished network, with no retraining. So this
          card follows the level and {FIRST_CLICK_POLICY_LABELS[policy].toLowerCase()} selection but not
          the density one, and the win rate shown is its Standard result. Open it for the full breakdown
          across all three densities.
        </p>
      </div>

      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="block w-full rounded-xl text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60"
      >
        <Card
          interactive
          className={cn("flex w-full items-center gap-4 transition-colors", open && "border-primary/50 shadow-md shadow-primary/10")}
          style={open ? { borderColor: `${accentColor}80` } : undefined}
        >
          <span
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full"
            style={{ backgroundColor: `${accentColor}1a`, color: accentColor }}
            aria-hidden="true"
          >
            <Trophy className="h-4 w-4" />
          </span>
          <div className="min-w-0 flex-1">
            <h3 className="font-semibold text-heading">
              {envPipeline?.label ??
                (best.runBrief.variant ? humanizeVariant(best.runBrief.variant) : best.runBrief.title)}
            </h3>
            <p className="mt-0.5 text-sm text-text-muted">
              {isTransfer
                ? `Trained under ${OTHER_ENV_LABEL[env]} — shown here as a generalization result`
                : "Best found model at this configuration"}
            </p>
          </div>
          <span className="hidden font-mono text-sm font-semibold tabular-nums sm:inline" style={{ color: accentColor }}>
            {formatPercent(best.runBrief.win_rate)}
          </span>
          <ChevronDown className={cn("h-4 w-4 shrink-0 text-text-muted transition-transform", open && "rotate-180 text-primary")} aria-hidden="true" />
        </Card>
      </button>

      {isTransfer && (
        <div className="flex items-start gap-2 rounded-lg border border-border px-3 py-2 text-xs text-text-muted">
          <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          <p>
            No {agentName} run has been trained under {FIRST_CLICK_POLICY_LABELS[policy].toLowerCase()} at this
            board size yet. The win rate and episode above are this {OTHER_ENV_LABEL[env]} model evaluated on
            these boards, so they measure how well it transfers — not how well it was trained to play here.
          </p>
        </div>
      )}

      {open && <VariantStoryCard story={best} about={about} isBest accentColor={accentColor} hideHeader />}
    </div>
  );
}
