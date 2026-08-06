import type { CSSProperties } from "react";
import { Trophy } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { HyperparameterTable, formatValue } from "@/components/experiment/HyperparameterTable";
import { deriveObservation, formatPercent, formatReward, humanizeVariant, type VariantStory } from "@/lib/experimentAdapters";
import { CONCEPT_GLOSSARY } from "@/lib/agentExplainers";
import { cn } from "@/lib/cn";

interface VariantStoryCardProps {
  story: VariantStory;
  /** The family's own baseline description -- shown as the baseline card's
   * "starting point" line, since it has nothing to diff against itself. */
  baselineDescription: string;
  isBest?: boolean;
  /** Synced with `VariantTimeline` -- true when this card's pill is selected. */
  isSelected?: boolean;
  accentColor: string;
}

/**
 * One training run rendered as a lab-notebook entry, in the five parts a
 * research log would have: why it was tested, what changed, its training
 * setup, its result, and what it taught us. Every field traces to
 * `VariantStory` (`@/lib/experimentAdapters`) -- nothing here is authored
 * per variant.
 */
export function VariantStoryCard({ story, baselineDescription, isBest, isSelected, accentColor }: VariantStoryCardProps) {
  const { runBrief, detail, isBaseline, addedTechniques, changes } = story;
  const hypothesisSentences = addedTechniques.map((t) => CONCEPT_GLOSSARY[t]).filter((s): s is string => Boolean(s));
  const observation = deriveObservation(story);

  return (
    <Card
      className={cn(
        "flex flex-col gap-4 border-white/10 bg-gradient-to-b from-surface/80 to-surface/50 shadow-lg shadow-black/[0.06] backdrop-blur-sm transition-shadow",
        isSelected && "ring-2 ring-offset-1 ring-offset-background",
      )}
      style={{
        borderLeft: `3px solid ${accentColor}`,
        ...(isSelected ? ({ "--tw-ring-color": accentColor } as CSSProperties) : {}),
      }}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="font-semibold text-heading">{runBrief.variant ? humanizeVariant(runBrief.variant) : runBrief.title}</h4>
        <div className="flex items-center gap-1.5">
          {isBaseline && <Badge variant="outline">Baseline</Badge>}
          {isBest && (
            <Badge variant="neutral" className="gap-1 text-primary">
              <Trophy className="h-3 w-3" />
              Best
            </Badge>
          )}
        </div>
      </div>

      <div>
        <p className="text-xs font-medium tracking-wide text-text-muted uppercase">
          {isBaseline ? "Starting point" : "Why was this tested?"}
        </p>
        {isBaseline ? (
          <p className="mt-1 text-sm text-text">{baselineDescription}</p>
        ) : hypothesisSentences.length > 0 ? (
          <ul className="mt-1 flex flex-col gap-1.5">
            {hypothesisSentences.map((sentence) => (
              <li key={sentence} className="text-sm text-text">
                {sentence}
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-1 text-sm text-text-muted">Tests a different {humanizeVariant(runBrief.variant ?? "")} configuration.</p>
        )}
      </div>

      {changes.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-medium tracking-wide text-text-muted uppercase">What changed?</p>
          <ul className="flex flex-wrap gap-1.5">
            {changes.map((change) => (
              <li
                key={change.key}
                className="rounded-full border border-border bg-surface-hover/60 px-2.5 py-1 font-mono text-xs text-text"
              >
                {change.key.replace(/_/g, " ")}: {formatValue(change.from)} → {formatValue(change.to)}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div>
        <p className="mb-1.5 text-xs font-medium tracking-wide text-text-muted uppercase">Training setup</p>
        <HyperparameterTable data={detail.hyperparameters} />
      </div>

      <div>
        <p className="mb-1.5 text-xs font-medium tracking-wide text-text-muted uppercase">Result</p>
        <dl className="flex flex-wrap items-center gap-6 text-sm">
          <div>
            <dt className="text-xs text-text-muted">Episodes</dt>
            <dd className="font-mono font-medium text-text">{runBrief.episodes.toLocaleString()}</dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">Win rate</dt>
            <dd className="font-mono font-medium text-text">{formatPercent(runBrief.win_rate)}</dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">Avg. reward</dt>
            <dd className="font-mono font-medium text-text">{formatReward(runBrief.avg_reward)}</dd>
          </div>
        </dl>
      </div>

      <div className="border-t border-border pt-3">
        <p className="text-xs font-medium tracking-wide text-text-muted uppercase">What did we learn?</p>
        <p className="mt-1 text-sm text-text">{observation}</p>
      </div>
    </Card>
  );
}
