import { ListChecks, Trophy } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { HyperparameterTable, formatValue } from "@/components/experiment/HyperparameterTable";
import { VariantDensityTable } from "@/components/research/VariantDensityTable";
import { VariantTrainingCurves } from "@/components/research/VariantTrainingCurves";
import { NarrativeText, type Narrative } from "@/components/research/NarrativeText";
import { deriveObservation, formatPercent, formatReward, humanizeVariant, type VariantStory } from "@/lib/experimentAdapters";

interface VariantStoryCardProps {
  story: VariantStory;
  /** What this variant is / why it was tested -- resolved once by
   * `ExperimentSetup`'s `deriveAbout` (hand-authored `VARIANT_NARRATIVES`
   * entry, or a mechanical fallback), so this and `VariantFlowCard`'s
   * teaser line never disagree. */
  about: string;
  /** What this variant actually taught us -- the hand-authored
   * `VARIANT_NARRATIVES` limitation/finding when one exists (real and
   * specific, e.g. "loss spikes unpredictably late in training"), or the
   * mechanical win-rate-delta sentence (`deriveObservation`) otherwise.
   * Without this override, the baseline card's own mechanical fallback is
   * just "starting point at X% win rate" -- true, but not a finding. */
  learned?: Narrative;
  isBest?: boolean;
  accentColor: string;
  /** Skips the title/badge row -- used by `VariantFlowCard`, whose own
   * collapsed header already shows the variant name and Baseline/Best
   * badges, so repeating them here would just be the same row twice. */
  hideHeader?: boolean;
}

/**
 * One training run rendered as a lab-notebook entry: why it was tested, what
 * changed, how it was trained/evaluated (mechanically from real
 * `ExperimentDetail` fields -- episodes, board, mines, eval episode count,
 * the same fixed seed every run in this project uses), its hyperparameters,
 * its training setup, its real training curves (`VariantTrainingCurves`,
 * collapsed/lazy -- makes visible what the prose elsewhere only describes,
 * e.g. baseline's loss spikes), its result (both the single stat row and,
 * where recovered, `VariantDensityTable`'s full sparse/standard/dense
 * comparison on the same 5x5 board), and what it taught us. Nothing here
 * beyond `about`/`learned` (both resolved upstream, see their docs) is
 * authored per variant -- everything else is mechanical derivation from
 * `VariantStory` (`@/lib/experimentAdapters`) and `ExperimentDetail`.
 */
export function VariantStoryCard({ story, about, learned, isBest, accentColor, hideHeader }: VariantStoryCardProps) {
  const { runBrief, detail, isBaseline, changes } = story;
  const observation = learned ?? deriveObservation(story);
  // Present for every run scored by the project's standard 2,000-episode
  // protocol; null only for an artifact that predates it being recorded.
  const ci = detail.evaluation_metrics.win_rate_ci95 ?? null;
  const evalEpisodes = detail.evaluation_metrics.eval_episodes ?? null;

  return (
    <Card
      className="flex flex-col gap-4 border-white/10 bg-gradient-to-b from-surface/80 to-surface/50 shadow-lg shadow-black/[0.06] backdrop-blur-sm"
      style={{ borderLeft: `3px solid ${accentColor}` }}
    >
      {!hideHeader && (
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
      )}

      <div>
        <p className="text-xs font-medium tracking-wide text-text-muted uppercase">
          {isBaseline ? "Starting point" : "Why was this tested?"}
        </p>
        <p className="mt-1 text-sm text-text">{about}</p>
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

      <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface-hover/40 px-4 py-3">
        <p className="flex items-center gap-2 text-xs font-medium tracking-wide text-text-muted uppercase">
          <ListChecks className="h-4 w-4 shrink-0" aria-hidden="true" />
          How it was trained and evaluated
        </p>
        <ul className="flex flex-col gap-1.5">
          <li className="flex gap-2.5 text-sm text-text">
            <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-text-muted" aria-hidden="true" />
            <span>
              Trained for {detail.episodes.toLocaleString()} episodes on a {detail.board} board ({detail.mines} mines).
            </span>
          </li>
          <li className="flex gap-2.5 text-sm text-text">
            <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-text-muted" aria-hidden="true" />
            <span>
              Evaluated over {(detail.evaluation_metrics.eval_episodes ?? 2000).toLocaleString()} greedy episodes
              (exploration off) at that same board size, on the same fixed seed (42) every agent in this project uses —
              so all of them face the identical set of boards.
            </span>
          </li>
        </ul>
      </div>

      <div>
        <p className="mb-1.5 text-xs font-medium tracking-wide text-text-muted uppercase">Hyperparameters</p>
        <HyperparameterTable data={detail.hyperparameters} />
      </div>

      <div>
        <p className="mb-1.5 text-xs font-medium tracking-wide text-text-muted uppercase">Training setup</p>
        <HyperparameterTable data={detail.training_configuration} />
      </div>

      <VariantTrainingCurves runId={runBrief.id} />

      <div>
        <p className="mb-1.5 text-xs font-medium tracking-wide text-text-muted uppercase">Result</p>
        <dl className="flex flex-wrap items-center gap-6 text-sm">
          <div>
            <dt className="text-xs text-text-muted">Episodes</dt>
            <dd className="font-mono font-medium text-text">{runBrief.episodes.toLocaleString()}</dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">Win rate</dt>
            <dd className="font-mono font-medium text-text">
              {formatPercent(runBrief.win_rate)}
              {ci ? (
                <span className="ml-1.5 font-normal text-xs text-text-muted">
                  95% CI {ci[0].toFixed(2)}–{ci[1].toFixed(2)}%
                </span>
              ) : null}
            </dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">Avg. reward</dt>
            <dd className="font-mono font-medium text-text">{formatReward(runBrief.avg_reward)}</dd>
          </div>
          <div>
            <dt className="text-xs text-text-muted">Eval episodes</dt>
            <dd className="font-mono font-medium text-text">
              {evalEpisodes?.toLocaleString() ?? "—"}
            </dd>
          </div>
        </dl>
        {/* Any run scored below the standard 2,000-episode protocol has too few
         * wins at these rates for its figure to be comparable -- say so rather
         * than letting it sit next to properly-scored numbers unmarked. */}
        {!ci && evalEpisodes != null && evalEpisodes < 1000 ? (
          <p className="mt-1.5 text-xs text-text-muted">
            Only {evalEpisodes.toLocaleString()} evaluation episodes — at a ~1–3% win rate that is a
            handful of wins, so treat this figure as indicative rather than comparable.
          </p>
        ) : null}
      </div>

      <VariantDensityTable
        runId={runBrief.id}
        standard={{
          win_rate: runBrief.win_rate ?? 0,
          avg_episode_length: detail.evaluation_metrics.avg_episode_length ?? 0,
          avg_reward: runBrief.avg_reward ?? 0,
        }}
        accentColor={accentColor}
      />

      <div className="border-t border-border pt-3">
        <p className="text-xs font-medium tracking-wide text-text-muted uppercase">What did we learn?</p>
        <NarrativeText value={observation} className="mt-1 text-sm text-text" />
      </div>
    </Card>
  );
}
