import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Quote, Sparkles } from "lucide-react";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { LevelDensitySelector } from "@/components/board/LevelDensitySelector";
import { ExperimentSetup } from "@/components/research/ExperimentSetup";
import { ExperimentComparison } from "@/components/research/ExperimentComparison";
import { getExperiment } from "@/api/experiments";
import { getExperimentMetrics, getLeaderboard } from "@/api/metrics";
import { useApiQuery } from "@/hooks/useApiQuery";
import { useBoardLevel } from "@/hooks/useBoardLevel";
import { slugifyAgentName } from "@/lib/agentAdapters";
import { isFamilySummary } from "@/lib/experimentAdapters";
import type { AgentKind } from "@/data/types";
import type { ExperimentDetail, ExperimentSummary } from "@/types/experiment";
import type { LeaderboardEntry, MetricsResponse } from "@/types/metrics";
import { cn } from "@/lib/cn";

interface ExperimentChamberProps {
  agentName: string;
  kind: AgentKind;
  accentColor: string;
  whyAttempted: string;
  researchQuestion: string;
  limitation: string;
  researchDecision: string;
  /** The next node's real tagline, or null for the last node (PPO). */
  nextTagline: string | null;
  leaderboardEntry: LeaderboardEntry | undefined;
}

interface ChamberData {
  detail: ExperimentDetail | null;
  metrics: MetricsResponse | null;
  family: ExperimentSummary | null;
  /** Full detail for every run in `family` (not just the leaderboard-linked
   * one) -- powers each variant's own hypothesis/changes/training-config
   * card in `ExperimentSetup`. Empty when there's no family. */
  variantDetails: ExperimentDetail[];
}

const CHAPTER_TITLES = ["Motivation", "Experiment Setup", "Experiment Comparison", "Research Decision"];

/** Mirrors `AgentDetail.tsx`'s `fetchAgentDetail` run-resolution, plus the old
 * `ExperimentDetail.tsx`'s family lookup -- both reused verbatim rather than
 * re-derived, just triggered lazily (on chamber open) instead of on page
 * load. Additionally resolves every sibling run's full detail so each
 * training variant can render as its own experiment card, not just the one
 * run the leaderboard happens to link to. */
async function fetchChamberData(experimentId: string | null): Promise<ChamberData> {
  if (!experimentId) return { detail: null, metrics: null, family: null, variantDetails: [] };

  const resolved = await getExperiment(experimentId);
  if (isFamilySummary(resolved)) return { detail: null, metrics: null, family: null, variantDetails: [] };

  const [metrics, familyResolved] = await Promise.all([
    getExperimentMetrics(experimentId),
    resolved.family_id ? getExperiment(resolved.family_id) : Promise.resolve(null),
  ]);

  const family = familyResolved && isFamilySummary(familyResolved) ? familyResolved : null;
  const variantDetails = family
    ? (await Promise.all(family.runs.map((run) => getExperiment(run.id)))).filter(
        (d): d is ExperimentDetail => !isFamilySummary(d),
      )
    : [];

  return { detail: resolved, metrics, family, variantDetails };
}

/**
 * One algorithm's research chapter: Motivation -> Experiment Setup ->
 * Experiment Comparison -> Research Decision. Every fact comes from real API
 * data (leaderboard + experiments); nothing here is invented per agent
 * beyond the hand-authored motivation/question/limitation/decision sentences
 * passed in from `ResearchPipeline`'s `STEPS`.
 */
export function ExperimentChamber({
  agentName,
  kind,
  accentColor,
  whyAttempted,
  researchQuestion,
  limitation,
  researchDecision,
  nextTagline,
  leaderboardEntry,
}: ExperimentChamberProps) {
  // This chamber's own level/density selection -- independent of every
  // other agent's chamber and of the pipeline row above (which always shows
  // the default board). At the default beginner/standard board, the
  // page-level `leaderboardEntry` prop (already fetched once for the whole
  // pipeline) is reused as-is rather than re-fetching the same data.
  const { configs, level, density, setLevel, setDensity } = useBoardLevel();
  const isDefaultLevel = level === "beginner" && density === "standard";

  const {
    data: scopedLeaderboard,
    status: scopedStatus,
    error: scopedError,
    isSlow: scopedSlow,
    retry: retryScoped,
  } = useApiQuery(() => (isDefaultLevel ? Promise.resolve(null) : getLeaderboard(level, density)), [level, density]);

  const effectiveEntry = isDefaultLevel ? leaderboardEntry : scopedLeaderboard?.find((entry) => entry.agent === agentName);

  const experimentId = effectiveEntry?.experiment_id ?? null;
  const { data, status, error, isSlow, retry } = useApiQuery(() => fetchChamberData(experimentId), [experimentId]);

  // Combines both fetches this chamber depends on -- the scoped leaderboard
  // lookup (only actually in flight at a non-default level/density) and the
  // chamber's own experiment data -- into one loading/error/retry surface,
  // so switching levels doesn't flash a stale "no data" state while the
  // scoped leaderboard entry is still resolving.
  const metricsLoading = status === "loading" || (!isDefaultLevel && scopedStatus === "loading");
  const metricsError = status === "error" ? error : !isDefaultLevel && scopedStatus === "error" ? scopedError : null;
  const metricsSlow = isSlow || (!isDefaultLevel && scopedSlow);
  function retryMetrics() {
    retry();
    if (!isDefaultLevel) retryScoped();
  }

  const slug = slugifyAgentName(agentName);
  const [activeChapter, setActiveChapter] = useState(0);
  const chapterRefs = useRef<(HTMLElement | null)[]>([]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length === 0) return;
        const topMost = visible.reduce((a, b) => (a.boundingClientRect.top < b.boundingClientRect.top ? a : b));
        const index = chapterRefs.current.findIndex((el) => el === topMost.target);
        if (index !== -1) setActiveChapter(index);
      },
      { rootMargin: "-15% 0px -70% 0px", threshold: 0 },
    );
    chapterRefs.current.forEach((el) => el && observer.observe(el));
    return () => observer.disconnect();
  }, [status]);

  function registerChapter(index: number) {
    return (el: HTMLElement | null) => {
      chapterRefs.current[index] = el;
    };
  }

  function scrollToChapter(index: number) {
    chapterRefs.current[index]?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <div className="flex gap-6">
      <div className="hidden shrink-0 flex-col items-center gap-3 pt-1.5 lg:flex">
        {CHAPTER_TITLES.map((title, index) => (
          <button
            key={title}
            type="button"
            onClick={() => scrollToChapter(index)}
            title={title}
            aria-label={`Jump to ${title}`}
            className="flex h-3 w-3 items-center justify-center rounded-full border transition-all"
            style={{
              borderColor: accentColor,
              backgroundColor: activeChapter === index ? accentColor : "transparent",
            }}
          />
        ))}
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-8">
        {configs.length > 0 && (
          <div className="flex flex-col gap-1.5">
            <span className="text-xs font-medium tracking-wide text-text-muted uppercase">{agentName}'s board</span>
            <LevelDensitySelector configs={configs} level={level} density={density} onLevelChange={setLevel} onDensityChange={setDensity} compact />
          </div>
        )}

        <Chapter index={1} title="Motivation" accentColor={accentColor} sectionRef={registerChapter(0)}>
          <p className="max-w-2xl text-sm text-text">{whyAttempted}</p>
          <div
            className="mt-3 flex max-w-2xl items-start gap-2.5 rounded-lg border bg-surface-hover/40 px-4 py-3"
            style={{ borderColor: `${accentColor}40` }}
          >
            <Quote className="mt-0.5 h-4 w-4 shrink-0" style={{ color: accentColor }} aria-hidden="true" />
            <div>
              <p className="text-xs font-medium tracking-wide text-text-muted uppercase">Research question</p>
              <p className="mt-1 text-sm font-medium text-heading">{researchQuestion}</p>
            </div>
          </div>
        </Chapter>

        {metricsLoading && (
          <div className="flex flex-col gap-3">
            <Skeleton className="h-48 w-full" />
            {metricsSlow && <ColdStartNotice />}
          </div>
        )}
        {!metricsLoading && metricsError && <ApiErrorState error={metricsError} onRetry={retryMetrics} title="Couldn't load this experiment" />}

        {!metricsLoading && !metricsError && data && (
          <>
            <Chapter index={2} title="Experiment Setup" accentColor={accentColor} sectionRef={registerChapter(1)}>
              <ExperimentSetup
                agentName={agentName}
                kind={kind}
                family={data.family}
                variantDetails={data.variantDetails}
                accentColor={accentColor}
              />
            </Chapter>

            <Chapter index={3} title="Experiment Comparison" accentColor={accentColor} sectionRef={registerChapter(2)}>
              <ExperimentComparison
                agentName={agentName}
                family={data.family}
                variantDetails={data.variantDetails}
                detail={data.detail}
                metrics={data.metrics}
                leaderboardEntry={effectiveEntry}
                accentColor={accentColor}
              />
            </Chapter>
          </>
        )}

        <Chapter index={4} title="Research Decision" accentColor={accentColor} sectionRef={registerChapter(3)}>
          <div className="flex flex-col gap-3">
            <p className="max-w-2xl text-sm font-medium text-heading">{researchDecision}</p>

            {limitation && (
              <div className="rounded-lg border border-amber-500/20 bg-amber-500/[0.06] px-4 py-3">
                <p className="text-xs font-medium tracking-wide text-amber-600 uppercase dark:text-amber-400">Limitation discovered</p>
                <p className="mt-1 text-sm text-text">{limitation}</p>
              </div>
            )}
            {nextTagline ? (
              <div className="rounded-lg border border-primary/20 bg-primary/[0.06] px-4 py-3">
                <p className="text-xs font-medium tracking-wide text-primary uppercase">What changed next</p>
                <p className="mt-1 text-sm text-text">{nextTagline}</p>
              </div>
            ) : (
              <div className="rounded-lg border border-primary/20 bg-primary/[0.06] px-4 py-3">
                <p className="flex items-center gap-2 text-sm text-text">
                  <Sparkles className="h-4 w-4 shrink-0" />
                  The current best-performing approach in this project.
                </p>
                <Link to="/agents" className="mt-2 flex items-center gap-1 text-sm font-medium text-primary hover:underline">
                  Browse every agent in detail
                  <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              </div>
            )}
          </div>
        </Chapter>

        <Link
          to={`/agents/${slug}`}
          className="flex w-fit items-center gap-1 text-sm font-medium text-primary hover:underline"
        >
          See {agentName} decide in real time
          <ArrowRight className="h-3.5 w-3.5" />
        </Link>
      </div>
    </div>
  );
}

/** One numbered section of an algorithm's research chapter -- a quiet
 * "01 / 02 / 03 / 04" marker plus title, so the chamber reads as a notebook
 * with a clear Problem -> Experiments -> Metrics -> Conclusion progression
 * rather than an unlabeled stack of cards. `sectionRef` feeds the
 * scroll-progress rail in `ExperimentChamber`. */
function Chapter({
  index,
  title,
  accentColor,
  sectionRef,
  children,
}: {
  index: number;
  title: string;
  accentColor: string;
  sectionRef: (el: HTMLElement | null) => void;
  children: ReactNode;
}) {
  return (
    <section ref={sectionRef} className={cn("flex scroll-mt-24 flex-col gap-3")}>
      <div className="flex items-center gap-2">
        <span className="font-mono text-xs font-semibold" style={{ color: accentColor }}>
          {String(index).padStart(2, "0")}
        </span>
        <h3 className="text-xs font-medium tracking-wide text-text-muted uppercase">{title}</h3>
        <span className="h-px flex-1 bg-border" aria-hidden="true" />
      </div>
      {children}
    </section>
  );
}
