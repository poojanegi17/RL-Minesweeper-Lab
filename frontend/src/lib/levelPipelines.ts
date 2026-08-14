import { getExperiment } from "@/api/experiments";
import { buildVariantStories, isFamilySummary, type VariantStory } from "@/lib/experimentAdapters";
import { type FirstClickPolicy } from "@/lib/boardLevelQuery";
import type { EnvVersion, ExperimentDetail, RunBrief } from "@/types/experiment";

export type PipelineLevel = "beginner" | "intermediate" | "expert";

export const LEVEL_LABELS_FULL: Record<PipelineLevel, string> = {
  beginner: "Beginner (5x5)",
  intermediate: "Intermediate (9x9)",
  expert: "Expert (16x16)",
};

/**
 * Which real experiment/family id backs each agent's per-level research
 * pipeline. Beginner is the original ablation family every DQN/PPO chamber
 * already had ("exp"/"ppo_exp"). Intermediate/Expert start as a single
 * standalone run (that level's carried-over best-5x5-config "baseline") and
 * become a real backend-grouped family once a second `_B_<variant>`-suffixed
 * sibling is trained alongside it (see `rl/evaluation/{dqn,ppo}_experiment.py`'s
 * `--output-dir results/{agent}_{level}_<LETTER>_<variant>` convention) --
 * update the id here to the bare family prefix (e.g. "ppo_intermediate")
 * once that happens, same as `ppo_intermediate` already is. Omitted entirely
 * for a level nobody has trained yet.
 */
export const LEVEL_PIPELINE_IDS: Record<string, Partial<Record<PipelineLevel, string>>> = {
  DQN: {
    beginner: "exp",
    intermediate: "dqn_intermediate",
    // Expert is omitted until a run exists under the current recipe. The
    // previous one predated every change in the Beginner pipeline and was
    // removed rather than shown alongside results it cannot be compared to.
  },
  PPO: {
    beginner: "ppo_exp",
    // Restored: the three runs previously here used ~7,400 gradient updates
    // against the ~150,000 the current pair uses, so they measured a starved
    // configuration rather than the agent. D/E replace them at a matched
    // budget and are what this level now reports.
    intermediate: "ppo_intermediate",
    // Expert still omitted -- no run exists under the current recipe.
  },
};

/**
 * Explicit run order for a level's list, overriding the backend's
 * id-alphabetical default (`exp_A`, `exp_B`, ...). That default is
 * chronological -- the order the runs happened to be made -- which for DQN
 * Beginner does not match the order that *explains* anything: `exp_K` was run
 * long after the four tuning arms but belongs immediately after the baseline
 * it extends, and the leave-one-out controls only make sense read against the
 * bundle they drop a factor from.
 *
 * So the order here follows the chapters rather than the filenames, and should
 * be kept in step with `PIPELINE_CHAPTERS` when a chapter is added or
 * reordered. A variant not listed keeps its backend position, after the
 * listed ones.
 */
export const VARIANT_DISPLAY_ORDER: Record<string, Partial<Record<PipelineLevel, string[]>>> = {
  DQN: {
    beginner: [
      // Chapter 1 -- the baseline and the four levers tried on it.
      "masked_target",
      "masked_lr_decay",
      "masked_shaped",
      "masked_slow_epsilon",
      "masked_deep",
      // Chapter 2 -- the same baseline, trained four times longer.
      "masked_longer",
      // Chapter 3 -- the three optimization fixes, each leave-one-out
      // control, then the architecture change on top of all three.
      "tuned",
      "no_reward_scale",
      "short_epsilon",
      "train_every_1",
      "fully_conv",
    ],
  },
};

/**
 * Variants kept on disk and served by the API, but not shown as pipeline cards.
 *
 * DQN Beginner's first five runs (the original baseline plus checkpoint, LR
 * decay, reduced capacity, and the two combined) all predate a correctness fix
 * to the bootstrap target: its argmax ranged over cells the agent can never
 * click, whose Q-values no gradient constrains, so the agent bootstrapped from
 * unconstrained numbers. Those runs measure an implementation that no longer
 * exists, which makes their premise invalid rather than their result merely
 * negative -- a reader taking "LR decay gives 3x" at face value learns
 * something untrue. The pipeline therefore starts from the corrected baseline
 * (`masked_target`), and each superseded arm was re-tested properly on top of
 * it:
 *
 *   - `checkpoint` showed the run destroys its own policy, which the baseline's
 *     training curve already shows on its own.
 *   - `lr_decay` was re-tested properly post-fix as `masked_lr_decay`.
 *   - `small_net` probed capacity, but under the bug, so it says little about
 *     the post-fix architecture question.
 *   - `combined` contributed one number -- the ~0.35pp run-to-run spread, since
 *     it was an accidental replicate of `lr_decay` -- which lives in the level
 *     conclusion's caveat rather than needing a card.
 *
 * They are hidden rather than deleted: the artifacts stay in `results_public/`,
 * `/api/experiments/exp` still returns all nine runs, and the README keeps the
 * full history. Remove an entry here to bring a card back.
 *
 * The rule this list applies, for whoever adds the next variant: **hide a
 * variant when its premise turned out to be invalid; keep it when the result is
 * valid but negative.** The four above assert things that are simply untrue
 * ("LR decay gives 3x"), so a reader who takes them at face value learns
 * something wrong. A null result is the opposite -- it is a true statement, and
 * usually a load-bearing one. `masked_shaped` and `masked_slow_epsilon` both
 * came back null and both stay visible, because the level conclusion's claim
 * that the reward signal and the exploration schedule are ruled out as the
 * binding constraint *is* those two runs. Dropping negative results because
 * they are unflattering is the publication bias the rest of this project goes
 * out of its way to avoid (the withdrawn 200-episode numbers, the accidental
 * replicate kept for its 0.35pp spread, two retired conclusions).
 */
export const HIDDEN_VARIANTS: Record<string, Partial<Record<PipelineLevel, string[]>>> = {
  DQN: {
    beginner: ["baseline", "checkpoint", "lr_decay", "small_net", "combined"],
  },
};

/** A level's stories filtered to the visible set and sorted into
 * `VARIANT_DISPLAY_ORDER`, leaving unlisted variants in their original relative
 * order at the end. */
export function visibleStories<T extends { runBrief: { variant?: string | null } }>(
  stories: T[],
  agentName: string,
  level: PipelineLevel,
): T[] {
  const hidden = new Set(HIDDEN_VARIANTS[agentName]?.[level] ?? []);
  const shown = hidden.size === 0 ? stories : stories.filter((s) => !hidden.has(s.runBrief.variant ?? ""));

  const order = VARIANT_DISPLAY_ORDER[agentName]?.[level];
  if (!order) return shown;
  const rank = (variant: string | null | undefined) => {
    const index = order.indexOf(variant ?? "");
    return index === -1 ? order.length : index;
  };
  return shown
    .map((story, index) => ({ story, index }))
    .sort((a, b) => rank(a.story.runBrief.variant) - rank(b.story.runBrief.variant) || a.index - b.index)
    .map(({ story }) => story);
}

export interface LevelPipelineData {
  stories: VariantStory[];
  bestRunId: string | null;
  /** The family's own mechanical description -- the baseline card's
   * "starting point" line when it has no hand-authored `about` override. */
  description: string;
}

/** Which board distribution a `FirstClickPolicy` selection corresponds to.
 * Mirrors `backend/app/board_levels.py`'s `FIRST_CLICK_POLICY_DIRS`. */
export const POLICY_ENV_VERSION: Record<FirstClickPolicy, EnvVersion> = { area: "v2", none: "v1" };

/**
 * The family describing one board distribution, when it isn't the level's own
 * `LEVEL_PIPELINE_IDS` entry.
 *
 * Retraining under `first_click_safe: area` produced a separate family rather
 * than another arm of the v1 ablation -- `dqn_v2_A_baseline` and the four
 * `ppo_v2_*` runs, the ones each agent's environment chapter is built on. They
 * are not in `exp`/`ppo_exp`, so a search restricted to the level's default
 * family finds no v2 run and concludes none was ever trained, when in fact the
 * best model either agent has at this board size is here. DQN Beginner is
 * exactly that case: 77.25% under v2, against the v1 family's 38.55%.
 *
 * Only Beginner needs an entry. DQN Intermediate keeps both distributions
 * inside one family (`dqn_intermediate_D_carried_config` at v1,
 * `..._E_env_v2` at v2), so `bestRunForEnv` already separates them there.
 *
 * A level with no entry for the selected env falls through to the default
 * family and the labelled-transfer path below -- which stays correct for
 * exactly the case it was written for: no run trained under that distribution
 * exists *anywhere*, not merely outside the default family.
 */
export interface EnvPipeline {
  /** Family or standalone-run id backing this distribution. */
  id: string;
  /** Card heading, for when the run's own variant name would misdescribe it.
   * `dqn_v2_A_baseline` is a standalone run, so `detailToRunBrief` labels it
   * "baseline" by construction -- which on a "best found model" card reads as
   * the *start* of a pipeline rather than its strongest result. */
  label?: string;
}

export const LEVEL_PIPELINE_ENV_IDS: Record<
  string,
  Partial<Record<PipelineLevel, Partial<Record<EnvVersion, EnvPipeline>>>>
> = {
  DQN: { beginner: { v2: { id: "dqn_v2_A_baseline", label: "Trained on first-click-safe boards" } } },
  PPO: { beginner: { v2: { id: "ppo_v2" } } },
};

/** The dedicated family for this `(agent, level, env)`, or `null` to use the
 * level's default one. */
export function pipelineIdForEnv(agentName: string, level: PipelineLevel, env: EnvVersion): EnvPipeline | null {
  return LEVEL_PIPELINE_ENV_IDS[agentName]?.[level]?.[env] ?? null;
}

/**
 * The best run *within one board distribution*, which is the only comparison
 * that means anything: a v1 and a v2 win rate measure different games, so the
 * family-wide `best_run_id` (a plain max over every run, computed in
 * `_metrics_summary`) will happily name a v2 run as "best" while the visitor is
 * looking at v1 results and a v1 replay.
 *
 * DQN Intermediate is exactly that case -- `dqn_intermediate_E_env_v2` scores
 * 80.15% under v2 and `dqn_intermediate_D_carried_config` 52.75% under v1, so
 * the max always picks E even under the "first click unsafe" toggle, where the
 * win rate shown and the episode played both come from D.
 *
 * When no run trained under `env` exists among `stories`, falls back to the
 * overall best and marks it `isTransfer` -- an honest description of what the
 * numbers beside it are: one distribution's checkpoint evaluated on the other's
 * boards, a generalization result rather than a matched one. Labelling that
 * beats hiding the card, which would drop a true account of the weights
 * actually being played; the caller is expected to surface the flag.
 *
 * The caller is also responsible for passing the right `stories`. This function
 * can only report on the runs it is given, so a family that happens to exclude
 * the env's real training run makes it claim a transfer that never happened --
 * see `LEVEL_PIPELINE_ENV_IDS`, which resolves that before the call.
 */
export function bestRunForEnv<T extends { runBrief: RunBrief }>(
  stories: T[],
  env: EnvVersion,
): { story: T; isTransfer: boolean } | null {
  const pickBest = (candidates: T[]): T => {
    const top = candidates.reduce((acc, story) => Math.max(acc, story.runBrief.win_rate ?? -1), -1);
    const tied = candidates.filter((story) => (story.runBrief.win_rate ?? -1) === top);

    if (tied.length === 1) return tied[0];

    // Ties are not a corner case here -- a whole level can sit at 0.00%. PPO
    // Intermediate has four runs all at exactly that. The API's own
    // `best_run_id` is a plain max and returns whichever the backend saw first
    // (the oldest, `..._D_carried_config`), so it used to be consulted here and
    // is deliberately no longer: taking it pointed "best model" at a superseded
    // run while the pipeline headlined its replacement. The max is recomputed
    // locally instead, which subsumes it whenever there is an outright winner.
    //
    // So among tied runs, prefer the most recent: `runs` arrives id-sorted and
    // this project names runs `_A_`, `_B_`, `_C_`... in the order they were
    // made (see `LEVEL_PIPELINE_IDS`), which makes the last id the latest
    // configuration rather than an arbitrary one.
    return tied[tied.length - 1];
  };

  const inEnv = stories.filter((story) => story.runBrief.env_version === env);
  if (inEnv.length > 0) return { story: pickBest(inEnv), isTransfer: false };
  if (stories.length === 0) return null;
  return { story: pickBest(stories), isTransfer: true };
}

function detailToRunBrief(detail: ExperimentDetail): RunBrief {
  return {
    id: detail.id,
    title: detail.title,
    // Every standalone level-pipeline run today *is* that level's baseline
    // (the previous level's best config, carried over unchanged) -- there's
    // no family-derived variant label to read for a single run, so this is
    // asserted by construction rather than parsed.
    variant: "baseline",
    episodes: detail.episodes,
    timestamp: detail.timestamp,
    win_rate: detail.evaluation_metrics.win_rate,
    avg_reward: detail.evaluation_metrics.avg_reward,
    metrics_available: detail.metrics_available,
    env_version: detail.env_version,
  };
}

/**
 * Resolves one level's pipeline data, whether `id` is a real multi-run
 * family (`ExperimentSummary`, e.g. `ppo_intermediate`'s baseline + high
 * entropy) or still just a single standalone run -- synthesized into the
 * exact same one-element `VariantStory[]` shape either way, so
 * `LevelPipeline` never needs to know which case it's in.
 */
export async function fetchLevelPipeline(id: string): Promise<LevelPipelineData> {
  const resolved = await getExperiment(id);

  if (isFamilySummary(resolved)) {
    const variantDetails = (await Promise.all(resolved.runs.map((run) => getExperiment(run.id)))).filter(
      (d): d is ExperimentDetail => !isFamilySummary(d),
    );
    return {
      stories: buildVariantStories(resolved.runs, variantDetails),
      bestRunId: resolved.metrics_summary.best_run_id,
      description: resolved.description,
    };
  }

  const runBrief = detailToRunBrief(resolved);
  return {
    stories: buildVariantStories([runBrief], [resolved]),
    bestRunId: resolved.id,
    description: resolved.description,
  };
}
