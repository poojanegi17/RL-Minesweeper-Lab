import { useState } from "react";
import { motion } from "framer-motion";
import { ChevronDown, FlaskConical, Clock } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { NarrativeText, type Narrative } from "@/components/research/NarrativeText";
import { VariantStoryCard } from "@/components/research/VariantStoryCard";
import { formatPercent, humanizeVariant, type VariantStory } from "@/lib/experimentAdapters";
import { runKey, type ChapterEvaluation, type ChapterTable, type PipelineChapter } from "@/lib/pipelineChapters";
import { cn } from "@/lib/cn";

interface ChapterFlowCardProps {
  chapter: PipelineChapter;
  /** This level's stories, keyed by variant -- the chapter resolves its own
   * headline and supporting rows out of these rather than being handed them,
   * so a variant missing from the API simply doesn't render a row. */
  storiesByVariant: Map<string, VariantStory>;
  /** The family this level page fetched -- the default for chapter rows that
   * don't name one of their own. */
  ownFamily: string;
  /** Per-variant hand-authored narrative, so an expanded run inside this
   * chapter reads exactly as it does in the "All runs" list rather than
   * falling back to the mechanical win-rate sentence. */
  narratives?: Record<string, { about: string; limitation: Narrative }>;
  isOpen: boolean;
  onClick: () => void;
  accentColor: string;
  index: number;
}

/** One run inside a chapter: the change it made, its win rate, and its full
 * lab-notebook entry behind a click. */
interface ChapterRun {
  story: VariantStory;
  /** What this run changed, in a few words. */
  change: string;
  /** Row heading. Defaults to the humanized variant, but a chapter can
   * override it -- a variant label only means something inside its own
   * family, so `ppo_long`'s `baseline` reads as "Baseline" next to
   * `ppo_exp`'s unless the chapter says what it actually is. */
  label: string;
  /** False when this run came from another family, in which case the level's
   * per-variant narratives do not describe it and must not be applied. */
  isOwnFamily: boolean;
}

/** Small table used for both a chapter's supporting runs and its hand-entered
 * tables. Scrolls horizontally on narrow screens rather than widening the
 * card, since the transfer table is five columns. */
function ChapterDataTable({
  caption,
  columns,
  rows,
  source,
  accentColor,
}: ChapterTable & { accentColor: string }) {
  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs font-medium tracking-wide text-text-muted uppercase">{caption}</p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[24rem] border-collapse text-sm">
          <thead>
            <tr className="border-b border-border">
              {columns.map((column, i) => (
                <th
                  key={column}
                  scope="col"
                  className={cn(
                    "py-2 pr-4 text-xs font-medium tracking-wide text-text-muted uppercase",
                    i === 0 ? "text-left" : "text-right",
                  )}
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label} className="border-b border-border/50 last:border-0">
                <th
                  scope="row"
                  className={cn(
                    "py-2 pr-4 text-left font-normal",
                    row.emphasis ? "font-medium text-heading" : "text-text",
                  )}
                  style={row.emphasis ? { color: accentColor } : undefined}
                >
                  {row.label}
                </th>
                {row.values.map((value, i) => (
                  <td
                    key={`${row.label}-${i}`}
                    className={cn(
                      "py-2 pr-4 text-right",
                      // A single wide value is prose (the "why" column of the
                      // setup chapter), so it reads left-aligned and unstyled;
                      // multiple values are figures.
                      row.values.length === 1 ? "text-left text-text-muted" : "font-mono tabular-nums text-text",
                      row.emphasis && row.values.length > 1 && "font-medium",
                    )}
                    style={row.emphasis && row.values.length > 1 ? { color: accentColor } : undefined}
                  >
                    {value}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-text-muted">{source}</p>
    </div>
  );
}

/** One evaluation of the chapter's checkpoint: collapsed to its conditions and
 * win rate, expanded to the full measurement and what it establishes. */
function EvaluationCard({
  evaluation,
  isOpen,
  onClick,
  accentColor,
}: {
  evaluation: ChapterEvaluation;
  isOpen: boolean;
  onClick: () => void;
  accentColor: string;
}) {
  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        onClick={onClick}
        aria-expanded={isOpen}
        className={cn(
          "flex w-full items-center gap-3 rounded-lg border px-3 py-2 text-left transition-colors",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60",
          isOpen ? "border-primary/40 bg-surface-hover" : "border-border hover:bg-surface-hover/60",
        )}
        style={isOpen ? { borderColor: `${accentColor}66` } : undefined}
      >
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-heading">{evaluation.label}</p>
          <p className="mt-0.5 text-xs text-text-muted">{evaluation.conditions}</p>
        </div>
        <span className="shrink-0 font-mono text-sm tabular-nums" style={{ color: accentColor }}>
          {(evaluation.winRate * 100).toFixed(2)}%
        </span>
        <ChevronDown
          className={cn("h-4 w-4 shrink-0 text-text-muted transition-transform", isOpen && "rotate-180 text-primary")}
          aria-hidden="true"
        />
      </button>

      {isOpen && (
        <Card className="flex flex-col gap-4">
          <dl className="flex flex-wrap gap-6 text-sm">
            <div>
              <dt className="text-xs text-text-muted">Win rate</dt>
              <dd className="font-mono font-medium text-text">{(evaluation.winRate * 100).toFixed(2)}%</dd>
            </div>
            <div>
              <dt className="text-xs text-text-muted">95% CI</dt>
              <dd className="font-mono font-medium text-text">
                {evaluation.ci95[0].toFixed(2)}–{evaluation.ci95[1].toFixed(2)}%
              </dd>
            </div>
            <div>
              <dt className="text-xs text-text-muted">Avg. episode length</dt>
              <dd className="font-mono font-medium text-text">{evaluation.avgEpisodeLength.toFixed(2)}</dd>
            </div>
            <div>
              <dt className="text-xs text-text-muted">Losses</dt>
              <dd className="font-mono font-medium text-text">{evaluation.losses}</dd>
            </div>
          </dl>
          <div>
            <h5 className="text-xs font-medium tracking-wide text-text-muted uppercase">What this shows</h5>
            <NarrativeText value={evaluation.finding} className="mt-1.5 text-sm text-text" />
          </div>
        </Card>
      )}
    </div>
  );
}

/**
 * One chapter of a level's story -- a *decision* rather than a single training
 * run. Collapsed it shows a title, a one-line teaser and its headline win
 * rate; expanded it shows what was tried, the runs backing that claim as a
 * compact table, and what the chapter establishes.
 *
 * The headline number comes from the API when the chapter names a `headline`
 * variant that resolved. Otherwise it falls back to the chapter's own
 * `staticWinRate` (real, measured, not yet mirrored into `results_public/`) or
 * renders as pending -- both badged, so a number the API did not serve is
 * never presented as though it did. See `pipelineChapters.ts`.
 */
export function ChapterFlowCard({ chapter, storiesByVariant, ownFamily, narratives, isOpen, onClick, accentColor, index }: ChapterFlowCardProps) {
  const [openRunId, setOpenRunId] = useState<string | null>(null);
  const headlineStory = chapter.headline
    ? storiesByVariant.get(runKey(chapter.headlineFamily ?? ownFamily, chapter.headline))
    : undefined;
  const winRate = headlineStory?.runBrief.win_rate ?? chapter.staticWinRate ?? null;
  const isPending = chapter.status === "pending" || winRate == null;
  // A chapter claims a `measured` number only while the API isn't serving it;
  // once the run is mirrored the headline resolves and the badge retires
  // itself without an edit here.
  const isUnpublished = !headlineStory && !isPending;

  // The chapter's runs, headline first: "one baseline and four tweaks" reads
  // as five comparable rows, not one number plus a separate table. A variant
  // the API didn't return is dropped rather than rendered blank.
  const headlineRun: ChapterRun[] = headlineStory
    ? [
        {
          story: headlineStory,
          change: chapter.headlineChange ?? "The run this chapter's siblings are measured against",
          label: chapter.headlineLabel ?? humanizeVariant(chapter.headline ?? ""),
          isOwnFamily: (chapter.headlineFamily ?? ownFamily) === ownFamily,
        },
      ]
    : [];

  const supportingRuns: ChapterRun[] = (chapter.supporting ?? []).flatMap((support) => {
    const family = support.family ?? ownFamily;
    const story = storiesByVariant.get(runKey(family, support.variant));
    return story
      ? [
          {
            story,
            change: support.change,
            label: support.label ?? humanizeVariant(support.variant),
            isOwnFamily: family === ownFamily,
          },
        ]
      : [];
  });

  // Ordered so the card reads the way the work went: a chapter whose siblings
  // are measured *against* its headline leads with it, one that builds *up* to
  // its headline ends with it.
  const runs: ChapterRun[] =
    chapter.headlinePosition === "last" ? [...supportingRuns, ...headlineRun] : [...headlineRun, ...supportingRuns];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.3 }}
      transition={{ duration: 0.35, ease: "easeOut", delay: index * 0.05 }}
      className="flex flex-col gap-3"
    >
      <button
        type="button"
        onClick={onClick}
        aria-expanded={isOpen}
        className="block w-full rounded-xl text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60"
      >
        <Card
          interactive
          className={cn("flex w-full items-center gap-4 transition-colors", isOpen && "border-primary/50 shadow-md shadow-primary/10")}
          style={isOpen ? { borderColor: `${accentColor}80` } : undefined}
        >
          <span
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full font-mono text-xs font-semibold"
            style={{ backgroundColor: `${accentColor}1a`, color: accentColor }}
            aria-hidden="true"
          >
            {index + 1}
          </span>

          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-1.5">
              <h4 className="font-semibold text-heading">{chapter.title}</h4>
              {isPending && (
                <Badge variant="outline" className="gap-1">
                  <Clock className="h-3 w-3" />
                  Not yet run
                </Badge>
              )}
              {isUnpublished && (
                <Badge variant="outline" className="gap-1">
                  <FlaskConical className="h-3 w-3" />
                  Measured
                </Badge>
              )}
            </div>
            <p className="mt-0.5 text-sm text-text-muted">
              {typeof chapter.about === "string" ? chapter.about : chapter.about[0]}
            </p>
          </div>

          <div className="flex shrink-0 items-center gap-3">
            <span className="hidden font-mono text-sm font-semibold tabular-nums sm:inline" style={{ color: isPending ? undefined : accentColor }}>
              {isPending ? "—" : formatPercent(winRate)}
            </span>
            <ChevronDown className={cn("h-4 w-4 text-text-muted transition-transform", isOpen && "rotate-180 text-primary")} aria-hidden="true" />
          </div>
        </Card>
      </button>

      {isOpen && (
        <Card className="flex flex-col gap-5">
          <div>
            <h5 className="text-xs font-medium tracking-wide text-text-muted uppercase">What was this?</h5>
            <NarrativeText value={chapter.about} className="mt-1.5 text-sm text-text" />
          </div>

          {isUnpublished && (
            <p className="rounded-lg border border-border bg-surface-hover px-4 py-3 text-xs text-text-muted">
              This run is evaluated and real, but hasn't been mirrored into the public results directory yet, so the
              figures below are entered here rather than served by the API.
            </p>
          )}

          {runs.length > 0 && (
            <div className="flex flex-col gap-2">
              <p className="text-xs font-medium tracking-wide text-text-muted uppercase">
                {chapter.supportingCaption ?? "Runs behind this"}
              </p>
              <div className="flex flex-col gap-2">
                {runs.map(({ story, change, label, isOwnFamily }) => {
                  const runId = story.runBrief.id;
                  const expanded = openRunId === runId;
                  // Narratives are keyed by variant within this level's family only.
                  const narrative = isOwnFamily ? narratives?.[story.runBrief.variant ?? ""] : undefined;

                  return (
                    <div key={runId} className="flex flex-col gap-2">
                      <button
                        type="button"
                        onClick={() => setOpenRunId(expanded ? null : runId)}
                        aria-expanded={expanded}
                        className={cn(
                          "flex w-full items-center gap-3 rounded-lg border px-3 py-2 text-left transition-colors",
                          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60",
                          expanded ? "border-primary/40 bg-surface-hover" : "border-border hover:bg-surface-hover/60",
                        )}
                        style={expanded ? { borderColor: `${accentColor}66` } : undefined}
                      >
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-sm font-medium text-heading">
                            {label}
                          </p>
                          <p className="mt-0.5 text-xs text-text-muted">{change}</p>
                        </div>
                        <span className="shrink-0 font-mono text-sm tabular-nums" style={{ color: accentColor }}>
                          {formatPercent(story.runBrief.win_rate)}
                        </span>
                        <ChevronDown
                          className={cn("h-4 w-4 shrink-0 text-text-muted transition-transform", expanded && "rotate-180 text-primary")}
                          aria-hidden="true"
                        />
                      </button>

                      {expanded && (
                        <VariantStoryCard
                          story={story}
                          about={narrative?.about ?? change}
                          learned={narrative?.limitation}
                          accentColor={accentColor}
                          hideHeader
                        />
                      )}
                    </div>
                  );
                })}
              </div>
              <p className="text-xs text-text-muted">Each run scored on the same 2,000 evaluation boards, seed 42.</p>
            </div>
          )}

          {chapter.table && <ChapterDataTable {...chapter.table} accentColor={accentColor} />}
          {chapter.transferTable && <ChapterDataTable {...chapter.transferTable} accentColor={accentColor} />}

          {chapter.evaluations && chapter.evaluations.length > 0 && (
            <div className="flex flex-col gap-2">
              <p className="text-xs font-medium tracking-wide text-text-muted uppercase">
                {chapter.evaluationsCaption ?? "Evaluations"}
              </p>
              <div className="flex flex-col gap-2">
                {chapter.evaluations.map((evaluation) => (
                  <EvaluationCard
                    key={evaluation.id}
                    evaluation={evaluation}
                    isOpen={openRunId === evaluation.id}
                    onClick={() => setOpenRunId(openRunId === evaluation.id ? null : evaluation.id)}
                    accentColor={accentColor}
                  />
                ))}
              </div>
              {chapter.evaluationsSource && <p className="text-xs text-text-muted">{chapter.evaluationsSource}</p>}
            </div>
          )}

          <div>
            <h5 className="text-xs font-medium tracking-wide text-text-muted uppercase">What did we learn?</h5>
            <NarrativeText value={chapter.limitation} className="mt-1.5 text-sm text-text" />
          </div>
        </Card>
      )}
    </motion.div>
  );
}
