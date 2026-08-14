import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Quote, Sparkles } from "lucide-react";
import { ExperimentSetup } from "@/components/research/ExperimentSetup";
import { NarrativeText, type Narrative } from "@/components/research/NarrativeText";
import { slugifyAgentName } from "@/lib/agentAdapters";
import type { AgentKind } from "@/data/types";
import { cn } from "@/lib/cn";

interface ExperimentChamberProps {
  agentName: string;
  kind: AgentKind;
  accentColor: string;
  whyAttempted: Narrative;
  researchQuestion: string;
  limitation: Narrative;
  researchDecision: Narrative;
  /** The next node's real tagline, or null for the last node (PPO). */
  nextTagline: string | null;
}

const CHAPTER_TITLES = ["Motivation", "Experiment Setup", "Research Decision"];

/**
 * One algorithm's research chapter: Motivation -> Experiment Setup ->
 * Research Decision. Every fact comes from real API data (leaderboard +
 * experiments -- `ExperimentSetup` and its `LevelPipeline`s resolve their
 * own data independently, this component no longer pre-fetches anything);
 * nothing here is invented per agent beyond the hand-authored
 * motivation/question/limitation/decision sentences passed in from
 * `ResearchPipeline`'s `STEPS`.
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
}: ExperimentChamberProps) {
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
        <Chapter index={1} title="Motivation" accentColor={accentColor} sectionRef={registerChapter(0)}>
          <NarrativeText value={whyAttempted} className="max-w-2xl text-sm text-text" />
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

        <Chapter index={2} title="Experiment Setup" accentColor={accentColor} sectionRef={registerChapter(1)}>
          <ExperimentSetup agentName={agentName} kind={kind} accentColor={accentColor} />
        </Chapter>

        <Chapter index={3} title="Research Decision" accentColor={accentColor} sectionRef={registerChapter(2)}>
          <div className="flex flex-col gap-3">
            <NarrativeText value={researchDecision} className="max-w-2xl text-sm font-medium text-heading" />

            {limitation && (
              <div className="rounded-lg border border-amber-500/20 bg-amber-500/[0.06] px-4 py-3">
                <p className="text-xs font-medium tracking-wide text-amber-600 uppercase dark:text-amber-400">Limitation discovered</p>
                <NarrativeText value={limitation} className="mt-1 text-sm text-text" />
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
