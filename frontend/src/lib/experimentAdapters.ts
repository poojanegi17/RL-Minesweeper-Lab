import { CONCEPT_GLOSSARY } from "@/lib/agentExplainers";
import type { ExperimentDetail, ExperimentSummary, RunBrief, SummaryValue } from "@/types/experiment";

/**
 * `GET /api/experiments/{id}` returns `ExperimentSummary` (a family, always
 * 2+ runs -- a standalone id resolves through the individual-run branch
 * instead, see backend `routes/experiments.py`) or `ExperimentDetail` (one
 * run). The two shapes share no discriminating field by name alone, so
 * narrow on `run_count`, which only `ExperimentSummary` declares.
 */
export function isFamilySummary(experiment: ExperimentSummary | ExperimentDetail): experiment is ExperimentSummary {
  return "run_count" in experiment;
}

/** The run `metrics_summary.best_run_id` points to, or null if no run in the
 * family recorded a win rate. Never guesses -- returns null rather than
 * falling back to an arbitrary run when the id doesn't resolve. */
export function bestRun(experiment: ExperimentSummary): RunBrief | null {
  const { best_run_id } = experiment.metrics_summary;
  if (best_run_id === null) return null;
  return experiment.runs.find((run) => run.id === best_run_id) ?? null;
}

export function formatPercent(value: number | null): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

export function formatReward(value: number | null): string {
  return value == null ? "—" : value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

/** "lr_decay" -> "LR Decay". Mirrors the backend's `humanize_variant` (used
 * server-side to build `RunBrief.title`) -- used here only for the standalone
 * variant chip, since `title` already includes the agent name. */
export function humanizeVariant(variant: string): string {
  return variant
    .split("_")
    .map((word) => (word.toLowerCase() === "lr" ? word.toUpperCase() : word.charAt(0).toUpperCase() + word.slice(1)))
    .join(" ");
}

export interface InvestigationNarrative {
  whyTested: string;
  whatChanged: string;
  whatLearned: string;
}

/**
 * Reframes a family's already-real, already-derived fields (`description`,
 * each run's `variant`, `techniques`, `metrics_summary`) as three research
 * questions instead of a flat artifact listing. Every sentence traces back
 * to a real field `GET /api/experiments` already returns -- nothing here is
 * authored per experiment; it's client-side synthesis of the same
 * mechanical data, in the same spirit as the backend's own
 * `derive_description`/`humanize_variant`. Returns `null` for a standalone
 * run (`run_count === 1`) -- there's no comparison to narrate for one run.
 */
export function buildInvestigationNarrative(experiment: ExperimentSummary): InvestigationNarrative | null {
  if (experiment.run_count <= 1) return null;

  const variantTitles = experiment.runs
    .map((run) => (run.variant ? humanizeVariant(run.variant) : null))
    .filter((title): title is string => title !== null);

  const whyTested = experiment.description;

  const whatChanged =
    variantTitles.length > 0
      ? `Each of the ${experiment.run_count} runs isolates a different change: ${variantTitles.join(", ")}.` +
        (experiment.techniques.length > 0 ? ` Techniques introduced across variants: ${experiment.techniques.join(", ")}.` : "")
      : `${experiment.run_count} runs recorded under this family, with no parseable per-run variant label.`;

  const best = bestRun(experiment);
  const { avg_win_rate: avgWinRate } = experiment.metrics_summary;
  let whatLearned: string;
  if (best && best.win_rate != null) {
    const baseline = experiment.runs.find((run) => run.variant === "baseline" && run.id !== best.id);
    if (baseline?.win_rate != null) {
      whatLearned = `${best.title} performed best at ${formatPercent(best.win_rate)} win rate, vs. ${formatPercent(baseline.win_rate)} for the baseline run.`;
    } else if (avgWinRate != null) {
      whatLearned = `${best.title} performed best at ${formatPercent(best.win_rate)} win rate, above the ${formatPercent(avgWinRate)} average across all runs.`;
    } else {
      whatLearned = `${best.title} performed best, at ${formatPercent(best.win_rate)} win rate.`;
    }
  } else {
    whatLearned = "No run in this investigation has recorded a win rate yet.";
  }

  return { whyTested, whatChanged, whatLearned };
}

const DIFF_EXCLUDED_KEYS = new Set(["train_seconds"]);

export interface VariantStory {
  runBrief: RunBrief;
  detail: ExperimentDetail;
  isBaseline: boolean;
  /** Techniques this run's `ExperimentDetail.techniques` has that the family's
   * baseline run doesn't -- a real set difference, not an authored list. */
  addedTechniques: string[];
  /** Field-level diff of `{...hyperparameters, ...training_configuration}`
   * against the same merged object on the baseline run. Empty for the
   * baseline itself. */
  changes: { key: string; from: SummaryValue; to: SummaryValue }[];
  /** `runBrief.win_rate - baseline.win_rate`. Null when either side is
   * missing, or this run *is* the baseline. */
  winRateDelta: number | null;
  /** `winRateDelta / baseline.win_rate * 100`. Null under the same
   * conditions as `winRateDelta`, plus when the baseline's win rate is 0
   * (division by zero, not a meaningful percentage). */
  winRateImprovementPct: number | null;
}

function mergedConfig(detail: ExperimentDetail): Record<string, SummaryValue> {
  return { ...detail.hyperparameters, ...detail.training_configuration };
}

/**
 * One story per real training run in a family -- "baseline" is whichever run
 * carries that exact `variant` label (falling back to the first run, the
 * same rule `buildInvestigationNarrative` uses for its own baseline
 * comparison). Everything else is diffed against it: `addedTechniques` and
 * `changes` are mechanical set/field differences over real `ExperimentDetail`
 * data, never authored prose -- callers turn `addedTechniques` into sentences
 * via `CONCEPT_GLOSSARY` (`@/lib/agentExplainers`), the same glossary
 * `AgentDetail` already uses to explain techniques.
 */
export function buildVariantStories(runs: RunBrief[], details: ExperimentDetail[]): VariantStory[] {
  const detailById = new Map(details.map((d) => [d.id, d]));
  const baselineRun = runs.find((r) => r.variant === "baseline") ?? runs[0];
  const baselineDetail = baselineRun ? detailById.get(baselineRun.id) : undefined;
  const baselineConfig = baselineDetail ? mergedConfig(baselineDetail) : {};
  const baselineTechniques = new Set(baselineDetail?.techniques ?? []);
  const baselineWinRate = baselineRun?.win_rate ?? null;

  return runs
    .map((runBrief): VariantStory | null => {
      const detail = detailById.get(runBrief.id);
      if (!detail) return null;

      const isBaseline = runBrief.id === baselineRun?.id;
      const addedTechniques = isBaseline ? [] : detail.techniques.filter((t) => !baselineTechniques.has(t));

      const config = mergedConfig(detail);
      const changes = isBaseline
        ? []
        : Object.keys(config)
            .filter((key) => !DIFF_EXCLUDED_KEYS.has(key))
            .filter((key) => JSON.stringify(config[key]) !== JSON.stringify(baselineConfig[key]))
            .map((key) => ({ key, from: baselineConfig[key] ?? null, to: config[key] }));

      const winRateDelta =
        isBaseline || runBrief.win_rate == null || baselineWinRate == null ? null : runBrief.win_rate - baselineWinRate;
      const winRateImprovementPct =
        winRateDelta == null || !baselineWinRate ? null : (winRateDelta / baselineWinRate) * 100;

      return { runBrief, detail, isBaseline, addedTechniques, changes, winRateDelta, winRateImprovementPct };
    })
    .filter((story): story is VariantStory => story !== null);
}

/**
 * "What did we learn?" for one variant card -- a mechanical sentence built
 * from `winRateDelta`/`winRateImprovementPct` alone, never authored per
 * variant. Baseline cards (nothing to compare against itself) get a neutral
 * framing instead.
 */
export function deriveObservation(story: VariantStory): string {
  const { runBrief, isBaseline, winRateDelta, winRateImprovementPct } = story;
  if (isBaseline) return `Starting point for this family at ${formatPercent(runBrief.win_rate)} win rate.`;
  if (winRateDelta == null) return "No baseline win rate recorded to compare this run against.";

  const from = formatPercent((runBrief.win_rate ?? 0) - winRateDelta);
  const to = formatPercent(runBrief.win_rate);
  const pct = winRateImprovementPct != null ? ` (${winRateImprovementPct >= 0 ? "+" : ""}${winRateImprovementPct.toFixed(0)}%)` : "";

  if (winRateDelta > 0) return `Win rate improved from ${from} to ${to}${pct} vs. baseline.`;
  if (winRateDelta < 0) return `Win rate fell from ${from} to ${to}${pct} vs. baseline.`;
  return `Win rate held steady at ${to} vs. baseline.`;
}

/**
 * "Why it won" for the best-performing run in a family -- the real technique
 * descriptions it added over the baseline (`CONCEPT_GLOSSARY`), joined; or
 * the family's own mechanical `whatLearned` sentence when it added no new
 * technique (e.g. the baseline itself happened to win). Never new prose.
 */
export function deriveWhyItWon(story: VariantStory, narrative: InvestigationNarrative | null): string {
  const sentences = story.addedTechniques.map((t) => CONCEPT_GLOSSARY[t]).filter((s): s is string => Boolean(s));
  if (sentences.length > 0) return sentences.join(" ");
  return narrative?.whatLearned ?? "";
}
