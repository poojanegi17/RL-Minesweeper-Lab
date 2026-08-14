import { useState } from "react";
import { ChevronDown, Trophy } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { VariantFlowCard } from "@/components/research/VariantFlowCard";
import { ChapterFlowCard } from "@/components/research/ChapterFlowCard";
import { LimitationConnector } from "@/components/research/LimitationConnector";
import { BoardSizeDensityTable } from "@/components/research/BoardSizeDensityTable";
import { useApiQuery } from "@/hooks/useApiQuery";
import { fetchLevelPipeline, LEVEL_LABELS_FULL, type PipelineLevel , visibleStories} from "@/lib/levelPipelines";
import { formatPercent, formatReward, humanizeVariant, type VariantStory } from "@/lib/experimentAdapters";
import { CONCEPT_GLOSSARY } from "@/lib/agentExplainers";
import { VARIANT_LEVEL_CONCLUSIONS, VARIANT_NARRATIVES } from "@/lib/variantNarratives";
import { chaptersFor } from "@/lib/pipelineChapters";
import { cn } from "@/lib/cn";

interface LevelPipelineProps {
  level: PipelineLevel;
  /** The real experiment/family id this level's pipeline is resolved from --
   * see `LEVEL_PIPELINE_IDS`. */
  levelId: string;
  agentName: string;
  accentColor: string;
}

/** `narrative.about` when this variant has a hand-authored entry; otherwise a
 * mechanical fallback (mirrors `ExperimentSetup`'s own, kept local here since
 * each level pipeline resolves its own family independently). */
function deriveAbout(story: VariantStory, baselineDescription: string, about: string | undefined): string {
  if (about) return about;
  if (story.isBaseline) return baselineDescription;
  const hypothesisSentences = story.addedTechniques.map((t) => CONCEPT_GLOSSARY[t]).filter((s): s is string => Boolean(s));
  if (hypothesisSentences.length > 0) return hypothesisSentences.join(" ");
  return `Tests a different ${humanizeVariant(story.runBrief.variant ?? "")} configuration.`;
}

/**
 * One board size's own research pipeline, collapsed behind a clickable
 * "Beginner" / "Intermediate" / "Expert" card -- like every other card in
 * this chamber, closed by default, fetched only once opened. Expanded, it
 * tells the same story the 5x5 ablation always has: each real variant tried
 * at this level as its own clickable card (`VariantFlowCard`), followed by
 * what was still wrong with it (`LimitationConnector`), until the last
 * variant tried *so far* closes with this level's real status
 * (`VARIANT_LEVEL_CONCLUSIONS`) -- "so far" because Intermediate/Expert
 * exploration is ongoing, not a closed investigation the way Beginner's is.
 * Ends with that level's best-found configuration and (for Intermediate/
 * Expert, where the sparse/dense numbers come from the same checkpoint as
 * the level's own result) how it holds up across mine density.
 */
export function LevelPipeline({ level, levelId, agentName, accentColor }: LevelPipelineProps) {
  const [isLevelOpen, setIsLevelOpen] = useState(false);
  const [openVariantId, setOpenVariantId] = useState<string | null>(null);
  const { data, status, error, isSlow, retry } = useApiQuery(
    () => (isLevelOpen ? fetchLevelPipeline(levelId) : Promise.resolve(null)),
    [levelId, isLevelOpen],
  );

  const narratives = VARIANT_NARRATIVES[agentName]?.[level];
  const conclusion = VARIANT_LEVEL_CONCLUSIONS[agentName]?.[level];

  return (
    <div className="flex flex-col gap-4">
      <button
        type="button"
        onClick={() => setIsLevelOpen((open) => !open)}
        aria-expanded={isLevelOpen}
        className="block w-full rounded-xl text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60"
      >
        <Card
          interactive
          className={cn("flex w-full items-center justify-between gap-4 transition-colors", isLevelOpen && "border-primary/50 shadow-md shadow-primary/10")}
          style={isLevelOpen ? { borderColor: `${accentColor}80` } : undefined}
        >
          <h3 className="font-semibold text-heading">{LEVEL_LABELS_FULL[level]}</h3>
          <ChevronDown className={cn("h-4 w-4 shrink-0 text-text-muted transition-transform", isLevelOpen && "rotate-180 text-primary")} aria-hidden="true" />
        </Card>
      </button>

      {isLevelOpen && (
        <div className="flex flex-col gap-4 pl-1">
          {status === "loading" && (
            <div className="flex flex-col gap-3">
              <Skeleton className="h-32 w-full" />
              {isSlow && <ColdStartNotice />}
            </div>
          )}
          {status === "error" && error && <ApiErrorState error={error} onRetry={retry} title="Couldn't load this level's pipeline" />}

          {status === "success" && data && (() => {
            // Chapter mode: this level's story is told as a few decisions
            // rather than one card per run, so the per-variant path below
            // (and with it `HIDDEN_VARIANTS`, whose hidden arms are chapter
            // 1's supporting rows) doesn't apply. The trailing "best
            // configuration" card is dropped too: it ranks whatever the API
            // serves, which would contradict a chapter reporting a real
            // result the API doesn't have yet. Each chapter closes itself.
            const chapters = chaptersFor(agentName, level);
            if (chapters) {
              const storiesByVariant = new Map(
                data.stories.flatMap((story) => (story.runBrief.variant ? [[story.runBrief.variant, story] as const] : [])),
              );
              return (
                <div className="flex flex-col">
                  {chapters.map((chapter, index) => (
                    <div key={chapter.id} className="flex flex-col">
                      <ChapterFlowCard
                        chapter={chapter}
                        storiesByVariant={storiesByVariant}
                        ownFamily={levelId}
                        isOpen={openVariantId === chapter.id}
                        onClick={() => setOpenVariantId(openVariantId === chapter.id ? null : chapter.id)}
                        accentColor={accentColor}
                        index={index}
                      />
                      <LimitationConnector
                        limitation={chapter.limitationBrief ?? chapter.limitation}
                        nextStep={chapter.nextStep}
                        conclusion={index === chapters.length - 1 ? conclusion : undefined}
                      />
                    </div>
                  ))}
                </div>
              );
            }

            const shown = visibleStories(data.stories, agentName, level);
            // Scoped to visible cards: the backend's best_run_id ranges over
            // every run in the family, including any hidden here, which would
            // otherwise badge or summarise a card nobody can see.
            const best = shown.reduce<(typeof shown)[number] | null>(
              (acc, story) =>
                story.runBrief.win_rate != null && (acc?.runBrief.win_rate == null || story.runBrief.win_rate > acc.runBrief.win_rate)
                  ? story
                  : acc,
              null,
            );
            const bestId = best?.runBrief.id ?? null;
            return (
            <>
              <div className="flex flex-col">
                {shown.map((story, index, ordered) => {
                  const narrative = narratives?.[story.runBrief.variant ?? ""];
                  const isLast = index === ordered.length - 1;

                  return (
                    <div key={story.runBrief.id} className="flex flex-col">
                      <VariantFlowCard
                        story={story}
                        about={deriveAbout(story, data.description, narrative?.about)}
                        learned={narrative?.limitation}
                        isBest={story.runBrief.id === bestId}
                        isOpen={openVariantId === story.runBrief.id}
                        onClick={() => setOpenVariantId(openVariantId === story.runBrief.id ? null : story.runBrief.id)}
                        accentColor={accentColor}
                        index={index}
                      />
                      {narrative?.limitation && (
                        <LimitationConnector
                          limitation={narrative.limitationBrief ?? narrative.limitation[0]}
                          nextStep={isLast ? undefined : narrative.nextStep}
                          conclusion={isLast ? conclusion : undefined}
                        />
                      )}
                    </div>
                  );
                })}
              </div>

              {(() => {
                const bestStory = best;
                if (!bestStory) return null;
                const bestLabel = bestStory.runBrief.variant ? humanizeVariant(bestStory.runBrief.variant) : bestStory.runBrief.title;

                return (
                  <>
                    <Card
                      className="border-white/10 bg-gradient-to-b from-primary/10 to-primary/[0.03] shadow-lg shadow-black/[0.06] backdrop-blur-sm"
                      style={{ borderTop: `3px solid ${accentColor}` }}
                    >
                      <div className="flex items-center gap-2" style={{ color: accentColor }}>
                        <Trophy className="h-4 w-4" />
                        <h4 className="text-xs font-medium tracking-wide uppercase">Best configuration so far</h4>
                      </div>
                      <p className="mt-2 text-lg font-semibold text-heading">
                        {agentName} — {bestLabel}
                      </p>
                      <dl className="mt-3 flex flex-wrap gap-6 text-sm">
                        <div>
                          <dt className="text-xs text-text-muted">Win rate</dt>
                          <dd className="font-mono font-medium text-text">{formatPercent(bestStory.runBrief.win_rate)}</dd>
                        </div>
                        {bestStory.winRateImprovementPct != null && (
                          <div>
                            <dt className="text-xs text-text-muted">Improvement over baseline</dt>
                            <dd className="font-mono font-medium text-emerald-500">
                              {bestStory.winRateImprovementPct >= 0 ? "+" : ""}
                              {bestStory.winRateImprovementPct.toFixed(0)}%
                            </dd>
                          </div>
                        )}
                        <div>
                          <dt className="text-xs text-text-muted">Avg. reward</dt>
                          <dd className="font-mono font-medium text-text">{formatReward(bestStory.runBrief.avg_reward)}</dd>
                        </div>
                      </dl>
                    </Card>

                    <BoardSizeDensityTable
                      level={level}
                      agentName={agentName}
                      standard={{
                        win_rate: bestStory.runBrief.win_rate ?? 0,
                        avg_episode_length: bestStory.detail.evaluation_metrics.avg_episode_length ?? 0,
                        avg_reward: bestStory.runBrief.avg_reward ?? 0,
                      }}
                      accentColor={accentColor}
                    />
                  </>
                );
              })()}
            </>
            );
          })()}
        </div>
      )}
    </div>
  );
}
