import { useState } from "react";
import { Trophy } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { VariantFlowCard } from "@/components/research/VariantFlowCard";
import { ChapterFlowCard } from "@/components/research/ChapterFlowCard";
import { LimitationConnector } from "@/components/research/LimitationConnector";
import { BoardSizeDensityTable } from "@/components/research/BoardSizeDensityTable";
import { useApiQuery } from "@/hooks/useApiQuery";
import { fetchLevelPipeline, type PipelineLevel, visibleStories } from "@/lib/levelPipelines";
import { formatPercent, formatReward, humanizeVariant, type VariantStory } from "@/lib/experimentAdapters";
import { CONCEPT_GLOSSARY } from "@/lib/agentExplainers";
import { VARIANT_LEVEL_CONCLUSIONS, VARIANT_NARRATIVES } from "@/lib/variantNarratives";
import { chaptersFor, extraFamiliesFor, runKey } from "@/lib/pipelineChapters";

interface LevelStoryProps {
  level: PipelineLevel;
  /** The real experiment/family id this level's pipeline resolves from --
   * see `LEVEL_PIPELINE_IDS`. */
  levelId: string;
  agentName: string;
  accentColor: string;
}

/** `narrative.about` when this variant has a hand-authored entry; otherwise a
 * mechanical fallback (mirrors `ExperimentSetup`'s own). */
function deriveAbout(story: VariantStory, baselineDescription: string, about: string | undefined): string {
  if (about) return about;
  if (story.isBaseline) return baselineDescription;
  const hypothesisSentences = story.addedTechniques.map((t) => CONCEPT_GLOSSARY[t]).filter((s): s is string => Boolean(s));
  if (hypothesisSentences.length > 0) return hypothesisSentences.join(" ");
  return `Tests a different ${humanizeVariant(story.runBrief.variant ?? "")} configuration.`;
}

/**
 * One board size's full research story, rendered at page scale rather than
 * inside a collapsed card -- this is what `/research/:agentSlug/:level`
 * shows.
 *
 * Two shapes, depending on the level. Where `pipelineChapters` defines a
 * chapter list, the story is those chapters -- the *decisions*, a handful of
 * them, each holding the runs that back it as expandable rows with their full
 * training curves and hyperparameter diffs. There is deliberately no separate
 * "all runs" list underneath: every run a chapter cites is reachable from
 * inside it, and a flat list of the same runs alongside only invited reading
 * them as peers of each other rather than as evidence for a decision.
 * Elsewhere it falls back to the original one-card-per-run flow, which is
 * still the right shape for a level with two or three runs and no
 * multi-phase story to tell.
 *
 * The chapter flow deliberately does *not* also render
 * `VARIANT_NARRATIVES`' per-variant connectors: a chapter already carries
 * its own limitation and next-step, and showing both would state the same
 * argument twice at two different altitudes.
 */
export function LevelStory({ level, levelId, agentName, accentColor }: LevelStoryProps) {
  const [openId, setOpenId] = useState<string | null>(null);

  const narratives = VARIANT_NARRATIVES[agentName]?.[level];
  const conclusion = VARIANT_LEVEL_CONCLUSIONS[agentName]?.[level];
  const chapters = chaptersFor(agentName, level);
  // A chapter may compare runs the backend groups into a different family
  // (PPO's budget chapter spans `ppo_exp` and `ppo_long`), so those are
  // fetched alongside this level's own. Joined into a string to keep the
  // effect dependency stable across renders.
  const extraFamilies = extraFamiliesFor(chapters, levelId);
  const extraKey = extraFamilies.join(",");

  const { data, status, error, isSlow, retry } = useApiQuery(async () => {
    const [own, ...extras] = await Promise.all([
      fetchLevelPipeline(levelId),
      ...extraFamilies.map(async (family) => [family, await fetchLevelPipeline(family)] as const),
    ]);
    return { own, extras: extras as (readonly [string, Awaited<ReturnType<typeof fetchLevelPipeline>>])[] };
  }, [levelId, extraKey]);

  if (status === "loading") {
    return (
      <div className="flex flex-col gap-3">
        <Skeleton className="h-64 w-full" />
        {isSlow && <ColdStartNotice />}
      </div>
    );
  }
  if (status === "error" && error) {
    return <ApiErrorState error={error} onRetry={retry} title="Couldn't load this level's pipeline" />;
  }
  if (status !== "success" || !data) return null;

  const own = data.own;

  // Chapter mode ranks nothing itself: the "best configuration" trailer reads
  // whatever the API serves, which would contradict a chapter reporting a
  // real result the API doesn't have yet (see `pipelineChapters`' header).
  const shown = visibleStories(own.stories, agentName, level);
  const best = shown.reduce<VariantStory | null>(
    (acc, story) =>
      story.runBrief.win_rate != null && (acc?.runBrief.win_rate == null || story.runBrief.win_rate > acc.runBrief.win_rate)
        ? story
        : acc,
    null,
  );

  if (chapters) {
    const storiesByVariant = new Map(
      [[levelId, own] as const, ...data.extras].flatMap(([family, pipeline]) =>
        pipeline.stories.flatMap((story) => (story.runBrief.variant ? [[runKey(family, story.runBrief.variant), story] as const] : [])),
      ),
    );

    return (
      <div className="flex flex-col">
        {chapters.map((chapter, index) => (
          <div key={chapter.id} className="flex flex-col">
            <ChapterFlowCard
              chapter={chapter}
              storiesByVariant={storiesByVariant}
              ownFamily={levelId}
              narratives={narratives}
              isOpen={openId === chapter.id}
              onClick={() => setOpenId(openId === chapter.id ? null : chapter.id)}
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

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col">
        {shown.map((story, index, ordered) => {
          const narrative = narratives?.[story.runBrief.variant ?? ""];
          const isLast = index === ordered.length - 1;

          return (
            <div key={story.runBrief.id} className="flex flex-col">
              <VariantFlowCard
                story={story}
                about={deriveAbout(story, own.description, narrative?.about)}
                learned={narrative?.limitation}
                isBest={story.runBrief.id === best?.runBrief.id}
                isOpen={openId === story.runBrief.id}
                onClick={() => setOpenId(openId === story.runBrief.id ? null : story.runBrief.id)}
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

      {best && (
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
              {agentName} — {best.runBrief.variant ? humanizeVariant(best.runBrief.variant) : best.runBrief.title}
            </p>
            <dl className="mt-3 flex flex-wrap gap-6 text-sm">
              <div>
                <dt className="text-xs text-text-muted">Win rate</dt>
                <dd className="font-mono font-medium text-text">{formatPercent(best.runBrief.win_rate)}</dd>
              </div>
              {best.winRateImprovementPct != null && (
                <div>
                  <dt className="text-xs text-text-muted">Improvement over baseline</dt>
                  <dd className="font-mono font-medium text-emerald-500">
                    {best.winRateImprovementPct >= 0 ? "+" : ""}
                    {best.winRateImprovementPct.toFixed(0)}%
                  </dd>
                </div>
              )}
              <div>
                <dt className="text-xs text-text-muted">Avg. reward</dt>
                <dd className="font-mono font-medium text-text">{formatReward(best.runBrief.avg_reward)}</dd>
              </div>
            </dl>
          </Card>

          <BoardSizeDensityTable
            level={level}
            agentName={agentName}
            standard={{
              win_rate: best.runBrief.win_rate ?? 0,
              avg_episode_length: best.detail.evaluation_metrics.avg_episode_length ?? 0,
              avg_reward: best.runBrief.avg_reward ?? 0,
            }}
            accentColor={accentColor}
          />
        </>
      )}
    </div>
  );
}
