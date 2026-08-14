import type { PipelineLevel } from "@/lib/levelPipelines";
import type { Narrative } from "@/components/research/NarrativeText";

/**
 * A level's story told as a handful of *decisions* rather than one card per
 * training run.
 *
 * DQN Beginner has eleven runs on disk. Five predate a correctness fix to the
 * bootstrap target and measure an implementation that no longer exists, so
 * `HIDDEN_VARIANTS` keeps them out of the run list entirely. The remaining six
 * are the narrative: a baseline, the four tuning levers measured against it,
 * and the longer run that showed the budget was the real constraint. Told one
 * card per run the four levers read as peers of the baseline, when their
 * collective result is a single finding -- so they are rows inside one chapter
 * rather than five cards.
 *
 * So a chapter has a headline run whose win rate is the card's number, and
 * optionally a `supporting` list of sibling runs shown as expandable rows
 * inside it. Every run stays reachable, in the chapter that cites it and in
 * "All runs" below; only the billing changes.
 *
 * Two chapters describe work that is planned or measured-but-unpublished
 * rather than served by the API, which is why `headline` is optional and
 * `status` exists. A chapter with no `headline` renders its number from
 * `staticWinRate` (measured, awaiting mirroring into `results_public/`) or as
 * "pending" (not yet run). Anything shown this way is badged in the UI, so the
 * page never presents an unproven number as though the API returned it. When
 * the corresponding run lands in `results_public/`, set `headline` to its
 * variant and delete `staticWinRate` -- the card then draws from real data
 * with no other change.
 */

/** One sibling run shown as a row inside a chapter's supporting table. */
export interface ChapterSupport {
  /** The real `RunBrief.variant` this row reads its win rate from. Rows whose
   * variant is missing from the API response are dropped rather than rendered
   * blank -- a chapter should degrade to its headline, not show holes. */
  variant: string;
  /** What this arm changed, in a few words. */
  change: string;
  /** Row heading, when the humanized variant would mislead. A variant label
   * only means something inside its own family -- `ppo_long`'s `baseline` is
   * a 100,000-episode run, not "the baseline". */
  label?: string;
  /** Experiment family this run lives in, when it isn't the level's own.
   *
   * A chapter usually draws from the family its level page fetched, but a
   * finding can span families -- PPO's budget chapter compares `ppo_exp`'s
   * 25,000-episode runs against `ppo_long`'s 100,000-episode ones, which the
   * backend groups separately because they were trained as separate series.
   * Naming the family here has `LevelStory` fetch it alongside the level's
   * own, so those rows expand to real training curves like any other rather
   * than being hand-entered into a table.
   *
   * Variant labels repeat across families (both PPO families have a
   * `baseline` and a `shaped`), so runs are keyed `family:variant`
   * internally -- never by variant alone. */
  family?: string;
}

/** One evaluation of a chapter's headline checkpoint, shown as its own
 * expandable card.
 *
 * Distinct from a `ChapterSupport` row on purpose: those are separate training
 * runs, each with its own history to chart. These are the *same* trained
 * weights scored under different board distributions, so they share one
 * training curve -- charting it once per card would repeat the identical graph
 * and imply three runs where there was one. Each card carries its own measured
 * numbers and what they establish instead. */
export interface ChapterEvaluation {
  id: string;
  /** What was scored, e.g. "Solvable boards, first click safe". */
  label: string;
  /** The board settings, verbatim, so the card states its own conditions. */
  conditions: string;
  winRate: number;
  ci95: [number, number];
  avgEpisodeLength: number;
  losses: number;
  /** What this particular evaluation establishes. */
  finding: Narrative;
}

/** A row of a chapter's own hand-entered table (measured outside the
 * experiments API -- e.g. a cross-benchmark evaluation of one checkpoint). */
export interface ChapterTableRow {
  label: string;
  values: string[];
  /** Renders this row in the accent colour -- for the row the card is about. */
  emphasis?: boolean;
}

export interface ChapterTable {
  caption: string;
  columns: string[];
  rows: ChapterTableRow[];
  /** Provenance line under the table, so a hand-entered table always says
   * where its numbers came from. */
  source: string;
}

export type ChapterStatus = "published" | "measured" | "pending";

export interface PipelineChapter {
  id: string;
  title: string;
  /** Collapsed teaser, and the expanded card's "What was this?" body. A point
   * list renders as bullets in both places; the collapsed teaser shows the
   * first point only, so lead with the sentence that stands alone. */
  about: Narrative;
  /** The real `RunBrief.variant` supplying this card's headline win rate.
   * Omitted for chapters the API doesn't serve yet (see `status`). */
  headline?: string;
  /** Family the headline run lives in, when it isn't the level's own -- same
   * rules as `ChapterSupport.family`. */
  headlineFamily?: string;
  /** Row heading for the headline run -- same reason as `ChapterSupport.label`. */
  headlineLabel?: string;
  /** Where the headline run sits among the rows. "first" (the default) suits a
   * chapter whose headline *is* the reference its siblings are measured
   * against. "last" suits one that builds up to its headline, where showing
   * the best result before the runs leading to it reads backwards. */
  headlinePosition?: "first" | "last";
  /** Win rate for a `measured` chapter -- real, evaluated, but not yet
   * mirrored into `results_public/`. Ignored when `headline` resolves. */
  staticWinRate?: number;
  status: ChapterStatus;
  /** What the headline run itself changed, shown on its own row above the
   * supporting ones. Only the first chapter's headline is a true "baseline";
   * every later chapter's is simply the run its siblings are measured
   * against, so the wording is per-chapter rather than assumed. */
  headlineChange?: string;
  /** Sibling runs backing this chapter's claim, shown as a table inside it. */
  supporting?: ChapterSupport[];
  /** Caption above the supporting table, stating what it collectively shows. */
  supportingCaption?: string;
  /** A hand-entered table (provenance required -- see `ChapterTable.source`). */
  table?: ChapterTable;
  /** A second hand-entered table, rendered below the first.
   *
   * Exists for the zero-shot transfer comparison at Intermediate, which cannot
   * be folded into `table`: that one is "this chapter's weights across mine
   * density", every row measured on a model trained at this board size, while
   * this one is "5x5 weights loaded here untouched, against the same recipe
   * trained here". Sharing a table would put trained and transferred numbers in
   * one column and read as a density effect, which is the exact confusion
   * `ChapterTable.source` exists to prevent. */
  transferTable?: ChapterTable;
  /** Evaluations of this chapter's headline checkpoint under different board
   * distributions, each its own expandable card. */
  evaluations?: ChapterEvaluation[];
  /** Heading above the evaluation cards. */
  evaluationsCaption?: string;
  /** Provenance line under the evaluation cards. */
  evaluationsSource?: string;
  /** What was still wrong / what this chapter establishes. */
  limitation: Narrative;
  /** One line for the connector into the next chapter. */
  limitationBrief?: string;
  /** What was tried next and which shortcoming it targeted. */
  nextStep?: string;
}

/**
 * DQN Beginner as four decisions: a baseline, the budget, the training setup,
 * and the board itself. Every number in the prose is on disk under
 * `rl/results_public/`, and the environment chapter's evaluation cards are
 * real 2,000-episode scorings of the deployed weights under the board settings
 * named on each card.
 *
 * The last chapter needs care when editing. Its headline 77.25% is measured on
 * a first-click-safe distribution, while chapters 1-3 are all measured on the
 * unprotected one -- so 38.55% and 77.25% must never be subtracted. That is
 * exactly why the chapter re-scores the previous chapter's agent on its own
 * benchmark (54.90%): the 22.35-point gain against *that* number is a
 * like-for-like comparison, and the only one the chapter claims.
 */
export const PIPELINE_CHAPTERS: Record<string, Partial<Record<PipelineLevel, PipelineChapter[]>>> = {
  DQN: {
    beginner: [
      {
        id: "dqn-baseline",
        title: "The DQN baseline",
        about: [
          "Double DQN over an 11-channel board encoding -- hidden mask, revealed mask, and a one-hot count of revealed neighbours.",
          "Experience replay and a target network synced every 25 episodes. 25,000 episodes at lr 1e-4, the default network, final weights deployed as-is.",
          "Nothing here is tuned. This is the reference every later change is measured against.",
          "On top of it we tried the four standard levers a value-based agent offers: decay the learning rate, stretch the exploration schedule, shape the reward for denser feedback, and give the network more depth.",
          "All five runs share a seed, so the only thing differing between them is the change named on each row.",
        ],
        headline: "masked_target",
        headlineLabel: "Baseline",
        headlineChange: "Double DQN at 25,000 episodes, nothing tuned",
        status: "published",
        supportingCaption:
          "The baseline and the four levers tried on top of it -- click any run for its full training curves and hyperparameters:",
        supporting: [
          { variant: "masked_lr_decay", label: "LR decay", change: "Step the learning rate down at 10k / 20k" },
          { variant: "masked_shaped", label: "Shaped reward", change: "Cascade bonus, sharper mine penalty" },
          { variant: "masked_slow_epsilon", label: "Slower exploration", change: "Stretch the epsilon decay schedule" },
          { variant: "masked_deep", label: "Deeper network", change: "Four convolutional layers instead of two" },
        ],
        limitation: [
          "The baseline reaches 11.40% -- well clear of Random's 0.45%, so the agent is genuinely learning something about the board.",
          "What the four levers helped with: very little. Decaying the learning rate was the best at 12.85% and stretching exploration reached 12.30%.",
          "None of them clears roughly 13% either -- every arm lands between 6.35% and 12.85%, which looked at the time like a ceiling.",
          "Reward shaping did nothing measurable -- 11.80% against 11.40%, p = 0.69. Whatever is binding, it is not that the reward signal is too sparse.",
          "A deeper network was clearly worse (6.35%), which rules out capacity as the answer before we go changing the architecture.",
          "It was not a ceiling. All five ran 25,000 episodes, and the next chapter shows the agent was still learning when they stopped -- so what these five arms actually measured was five different ways of not training long enough.",
        ],
        limitationBrief:
          "11.40% to start, and none of the four standard levers moves it past roughly 13%.",
        nextStep:
          "Four levers all landing in the same narrow band is suspicious -- it suggests none of them was the binding constraint. Before changing the agent any further, the cheapest thing to rule out is whether 25,000 episodes was simply too few.",
      },
      {
        id: "budget",
        title: "The ceiling was the budget",
        about: [
          "The baseline configuration again, unchanged in every respect except one: 100,000 episodes instead of 25,000.",
          "Same seed, same network, same learning rate, same exploration schedule, same final-weights deployment, scored on the same 2,000 boards.",
          "Nothing about the agent is different, so whatever this run gains is attributable to training length alone.",
        ],
        headline: "masked_longer",
        headlineLabel: "100,000 episodes",
        headlineChange: "The baseline configuration, trained four times longer",
        status: "published",
        supportingCaption:
          "Click for its full training curves and hyperparameters:",
        limitation: [
          "21.25%, against the baseline's 11.40% -- the same agent, nearly doubled, by nothing but more episodes (p < 0.00001).",
          "So the band those four levers landed in was never a ceiling. At 25,000 episodes the agent was still climbing steeply, and every arm was measuring a run stopped too early.",
          "The training curve makes it plain: about 2% over the first 10,000 episodes, 11% by 30,000, 16% by 50,000.",
          "This time the plateau is real. The last 50,000 episodes sit between 17% and 19% with no upward trend, so a longer run would buy little -- unlike the previous one, that claim now has the evidence behind it.",
          "It also reframes the four tuning arms. They were not wrong about their levers; they were asking whether a hyperparameter helps while the dominant variable was still unspent.",
        ],
        limitationBrief:
          "Four times the training nearly doubles it -- 11.40% to 21.25%. The apparent ceiling was an unfinished run.",
        nextStep:
          "With the budget spent and the curve genuinely flat, the remaining suspects are the things no run so far has varied: how the loss sees the reward, how often the agent trains on what it collects, and an architecture whose Linear head is 94% of its parameters.",
      },
      {
        id: "training-setup",
        title: "Three fixes that only work together",
        about: [
          "With the budget spent, three settings had still never been varied.",
          "How the loss sees the reward (`reward_scale`), how long exploration lasts (`epsilon_decay`), and how often the agent trains on what it collects (`train_every`).",
          "We built up from the previous chapter's 21.25% rather than guessing which mattered: every pairing of the three first, then all three together, then the conv head on top.",
          "The deltas between rows are what each change is worth once the other two are already in place.",
        ],
        headline: "fully_conv",
        headlineLabel: "All three, plus the conv head",
        headlineChange: "fully_conv on top of the three fixes -- 29,089 parameters at any board size, against 111,993",
        headlinePosition: "last",
        status: "published",
        supportingCaption:
          "Building up: each pairing of the three fixes, then all three, then the conv head -- click any run for its full training curves and hyperparameters:",
        supporting: [
          {
            variant: "train_every_1",
            label: "Reward scaling + slower exploration",
            change: "Both, with the replay ratio left at train_every 1 -- and the worst run in this chapter",
          },
          {
            variant: "no_reward_scale",
            label: "Reduced replay ratio + slower exploration",
            change: "Both, with reward_scale left at 1.0",
          },
          {
            variant: "short_epsilon",
            label: "Reduced replay ratio + reward scaling",
            change: "Both, with epsilon_decay left at the original 0.995",
          },
          {
            variant: "tuned",
            label: "All three fixes",
            change: "reward_scale 0.1, train_every 4 and epsilon_decay 0.9997 together, on the default network",
          },
        ],
        table: {
          caption: "The chapter's best weights, scored across every mine density",
          columns: ["Density", "Mines", "Of the board", "Win rate", "95% CI"],
          rows: [
            { label: "Sparse", values: ["3", "12.0%", "70.35%", "68.35 - 72.35"] },
            { label: "Standard", values: ["5", "20.0%", "38.55%", "36.42 - 40.68"], emphasis: true },
            { label: "Dense", values: ["8", "32.0%", "3.30%", "2.52 - 4.08"] },
          ],
          source:
            "2,000 greedy episodes per density, seed 42, scoring the deployed weights on unprotected boards. Trained once at standard density and evaluated at the other two without retraining.",
        },
        limitation: [
          "All three together take the agent from 21.25% to 37.90% -- 16.65 points, and the largest step in this pipeline so far (the budget was worth 9.85).",
          "Adding the reduced replay ratio to the other two is worth 31.95 points, by far the biggest of the three: 5.95% to 37.90%.",
          "Adding reward scaling to the other two is worth 9.95 points (27.95% to 37.90%), and adding the longer exploration schedule 6.95 points (30.95% to 37.90%). Both large, both highly significant.",
          "Those three gains sum to 48.85 points, against a bundle worth 16.65. They cannot be added, because they are not independent -- each is worth far more when the other two are already in place.",
          "The clearest sign of that is the first row. Reward scaling and slower exploration together, without the reduced replay ratio, score 5.95% -- far *worse* than applying none of the three. Two of these changes are actively harmful on their own.",
          "The mechanism is visible in that run's curve: it peaks near 7% around episode 40,000 and then decays for the rest of training, with Q-values staying bounded throughout. That is over-training on a replay buffer, not the divergence pattern seen elsewhere.",
          "Adding the conv head on top is the honest disappointment: 38.55% against 37.90%, a 0.65-point difference that is not significant (p = 0.67). At 5x5 it buys no accuracy at all.",
          "It earns its place structurally rather than numerically -- 29,089 parameters instead of 111,993, and identical at every board size, so one model can be loaded at 9x9 or 16x16 instead of retrained. That is a transfer argument, and it should not be presented as a win-rate one.",
          "Across density this configuration tracks the CSP solver at a roughly constant deficit until the board gets crowded: 70.35% against CSP's 74.45% at 3 mines, 38.55% against 43.40% at 5, then 3.30% against 10.75% at 8.",
          "Dense is where it stops being a near-miss and becomes a different outcome. Episodes there average 3.81 moves against 6.70 at standard -- it is dying almost immediately rather than losing late.",
        ],
        limitationBrief:
          "16.65 points from three fixes -- but only together; two of them make things worse alone. The conv head adds nothing measurable at 5x5.",
        nextStep:
          "At 38.55% the agent is well past where tuning left it and still short of the CSP solver's 43.40% on the same boards. Every change so far has been to the agent. The one thing never varied is the game it is being asked to play.",
      },
      {
        id: "environment",
        title: "Changing the game, not the agent",
        about: [
          "Every change so far has been to the agent. This chapter changes the board instead, and leaves the agent exactly as the previous chapter left it -- fully_conv, all three fixes, same seed, same budget.",
          "`guarantee_solvable` resamples layouts until the board is clearable by deduction alone, so every position has a provably correct move and a loss reflects an agent error rather than bad luck. Training only.",
          "`first_click_safe: area` keeps the whole 3x3 block around the opening click mine-free, so no game is lost on move one and every episode starts from a cascade.",
          "That second one reaches evaluation too, because it defines the benchmark rather than shaping training -- so the previous chapter's agent was re-scored on this benchmark as well, and the two can be compared on equal terms.",
        ],
        headline: "baseline",
        headlineFamily: "dqn_v2_A_baseline",
        headlineLabel: "Trained on solvable, first-click-safe boards",
        headlineChange: "fully_conv and all three fixes unchanged -- only the board generation differs",
        status: "published",
        supportingCaption:
          "The training run behind this chapter -- click for its full training curves and hyperparameters:",
        table: {
          caption: "The chapter's weights, scored across every mine density",
          columns: ["Density", "Mines", "Of the board", "Win rate", "95% CI"],
          rows: [
            { label: "Sparse", values: ["3", "12.0%", "89.40%", "88.05 - 90.75"] },
            { label: "Standard", values: ["5", "20.0%", "77.25%", "75.41 - 79.09"], emphasis: true },
            { label: "Dense", values: ["8", "32.0%", "38.90%", "36.76 - 41.04"] },
          ],
          source:
            "2,000 greedy episodes per density, seed 42, on first-click-safe boards with solvability unfiltered. Trained once at standard density and evaluated at the other two without retraining. Not comparable to the previous chapter's table, which is measured on unprotected boards.",
        },
        evaluationsCaption: "Scored on one benchmark: first click safe, boards unfiltered",
        evaluationsSource:
          "2,000 episodes each, seed 42, scoring the weights each run actually deployed. The first two differ only in the boards they trained on; the third changes the evaluation instead.",
        evaluations: [
          {
            id: "eval-previous-agent",
            label: "Previous chapter's agent, on this benchmark",
            conditions: "Trained on unprotected boards, scored with the first click safe",
            winRate: 0.549,
            ci95: [52.72, 57.08],
            avgEpisodeLength: 7.97,
            losses: 902,
            finding: [
              "54.90%. This is the previous chapter's agent, unchanged, simply scored on the new benchmark.",
              "It never saw a protected opening in training and still gains a great deal from one: 38.55% on the boards it trained on, 54.90% here.",
              "That 16.35-point jump is the benchmark's own contribution -- the same weights, playing an easier game. Any headline gain in this chapter has to be read net of it.",
              "It is also the row that makes the comparison below meaningful, because it puts both agents on identical boards.",
            ],
          },
          {
            id: "eval-benchmark",
            label: "This chapter's agent, on the same benchmark",
            conditions: "Trained on solvable, first-click-safe boards; scored on the same boards as above",
            winRate: 0.7725,
            ci95: [75.41, 79.09],
            avgEpisodeLength: 5.18,
            losses: 455,
            finding: [
              "77.25% against 54.90% for the previous agent, on identical boards -- a 22.35-point gain whose confidence intervals do not overlap.",
              "This is the chapter's real result. Both agents are architecturally identical and equally trained; the only difference is the distribution of boards they saw.",
              "Training on no-guess boards means every loss during training was the agent's own fault, so the learning signal is never contaminated by positions that could not have been played better.",
              "It also finishes games faster -- 5.18 moves per episode against 7.97 -- which is what you would expect from a policy that hesitates less because it was trained where hesitation was never rewarded.",
            ],
          },
          {
            id: "eval-trained-on",
            label: "The ceiling, on its own training distribution",
            conditions: "Solvable boards, first click safe -- the exact distribution it trained on",
            winRate: 0.9965,
            ci95: [99.39, 99.91],
            avgEpisodeLength: 4.61,
            losses: 7,
            finding: [
              "99.65% -- seven losses in two thousand games on boards where a correct move always exists.",
              "Nothing about the method prevents near-perfect play once the board is guaranteed solvable, which is worth knowing as an upper bound.",
              "It is also the least impressive of the three, because a deduction solver clears no-guess boards 100% of the time by construction. Matching that is table stakes.",
              "Its value is diagnostic: with 7 losses here against 455 on unfiltered boards, essentially all of this agent's benchmark losses come from positions no amount of deduction could have settled.",
            ],
          },
        ],
        limitation: [
          "On a fixed benchmark, training on solvable, first-click-safe boards is worth 22.35 points -- 54.90% to 77.25%, with non-overlapping confidence intervals. Not one line of the agent changed.",
          "That is the largest gain in this pipeline from a single change, and the only one that came from altering the problem rather than the learner.",
          "It has to be read alongside the benchmark's own contribution. The previous agent gains 16.35 points from the protected opening by itself, so a protected board is simply an easier game before any training difference is considered.",
          "The two board settings moved together, so the 22.35 points belong to the pair. Separating the no-guess curriculum from the safe opening needs one further run with only the opening made safe.",
          "The ceiling is not the constraint. At 99.65% on solvable boards this configuration can already play near-perfectly; what limits it on the benchmark is positions where no correct move exists.",
          "Across density it beats the CSP solver at two of three settings on these boards: 89.40% against CSP's 91.10% at 3 mines, 77.25% against 70.35% at 5, and 38.90% against 36.45% at 8.",
          "That is the reverse of the previous chapter, where the deficit to CSP widened as the board filled up. Trained on no-guess boards, this agent holds its own exactly where deduction runs short of information.",
          "A caveat on comparability: this run deployed best_model.pt while the previous chapter's runs deployed final weights, so checkpoint selection differs alongside the board distribution.",
        ],
        limitationBrief:
          "22.35 points from changing the boards rather than the agent -- though the protected benchmark is worth 16.35 of its own before any training difference.",
      },
    ],
    intermediate: [
      {
        id: "intermediate-carried",
        title: "The 5x5 recipe, carried to 9x9",
        about: [
          "The configuration that ended the Beginner pipeline, moved to a 9x9 board with 12 mines and retrained from scratch.",
          "Nothing about the agent is re-tuned: fully_conv, reward_scale 0.1, train_every 4, epsilon_decay 0.9997, 100,000 episodes, same seed.",
          "This is only possible because of the conv head. The old Linear head was board-size-specific, so a 5x5 model could not even be built at 9x9 -- the architecture change that bought no accuracy at Beginner is what makes this chapter exist.",
          "Boards are the original unprotected kind, where the opening click can lose outright, so this is directly comparable to the 38.55% the same recipe scored at 5x5.",
        ],
        headline: "carried_config",
        headlineLabel: "9x9, 12 mines, carried configuration",
        headlineChange: "The Beginner recipe unchanged, retrained at the larger board size",
        status: "published",
        supportingCaption:
          "The run behind this chapter -- click for its full training curves and hyperparameters:",
        table: {
          caption: "The same weights, scored across every mine density at this board size",
          columns: ["Density", "Mines", "Of the board", "Win rate", "95% CI"],
          rows: [
            { label: "Sparse", values: ["8", "9.9%", "78.90%", "77.11 - 80.69"] },
            { label: "Standard", values: ["12", "14.8%", "52.75%", "50.56 - 54.94"], emphasis: true },
            { label: "Dense", values: ["18", "22.2%", "8.00%", "6.81 - 9.19"] },
          ],
          source:
            "2,000 greedy episodes per density, seed 42, scoring the deployed weights on unprotected boards. Trained once at standard density and evaluated at the other two without retraining.",
        },
        transferTable: {
          caption: "Zero-shot transfer: the 5x5 weights loaded here untouched, against this chapter's trained model",
          columns: ["Density", "Transferred from 5x5", "Trained here", "Δ"],
          rows: [
            { label: "Sparse (8)", values: ["53.00%", "78.90%", "-25.90"] },
            { label: "Standard (12)", values: ["26.75%", "52.75%", "-26.00"], emphasis: true },
            { label: "Dense (18)", values: ["2.05%", "8.00%", "-5.95"] },
          ],
          source:
            "2,000 greedy episodes per cell, seed 42, on unprotected boards. `exp_M_fully_conv` loaded at 9x9 with no retraining -- possible only because the conv head removed the board-size dependence. The transfer column is a generalization result, not a matched one.",
        },
        limitation: [
          "52.75%, against 38.55% for the same recipe at 5x5. The bigger board is not harder for this agent -- it is easier.",
          "Mine density explains most of that: 12 mines in 81 cells is 14.8%, against 5 in 25 at 20.0%. More cells also means more of the board is constrained by revealed numbers at any moment.",
          "Episodes run far longer -- 16.60 moves on average against 6.70 -- so a win here requires sustaining correct play roughly two and a half times as long.",
          "The training curve climbs to about 31% by episode 40,000 and then flattens, so 100,000 episodes was enough and more would buy little.",
          "Density decides everything, and it decides it sharply. The same weights score 78.90% at 8 mines, 52.75% at 12 and 8.00% at 18 -- with no retraining between them.",
          "That is not a gentle slope. Going from 14.8% to 22.2% mine density costs 44.75 points, while the equivalent step down gains 26.15.",
          "Against the CSP solver on the same boards it holds up at low density and collapses at high: 78.90% against CSP's 81.45% at sparse, 52.75% against 60.90% at standard, and 8.00% against 23.10% at dense.",
          "The dense column also fails differently. Episodes there average 11.31 moves against 16.60 at standard -- it is not grinding out long games and losing at the end, it is dying early.",
          "For context, the three earlier attempts at this board -- 25,000 episodes on the old default network -- scored 0.00%, 0.05% and 0.05%. This is not an incremental improvement on them.",
          "Those three are excluded from this page. They predate every change in the Beginner pipeline and measure a configuration that no longer exists.",
        ],
        limitationBrief:
          "52.75% at 9x9 against 38.55% at 5x5 -- the recipe transfers, but density decides everything: 78.90% at 8 mines, 8.00% at 18.",
        nextStep:
          "At Beginner the board settings were worth more than anything done to the agent. The same two changes are the obvious next test here, on a board where episodes are long enough that a single forced guess is far likelier to end a run before it finishes.",
      },
      {
        id: "intermediate-environment",
        title: "The same board changes, at 9x9",
        about: [
          "The identical configuration retrained with the two board-generation settings that mattered most at Beginner.",
          "`first_click_safe: area` opens a mine-free 3x3 block on the first move; `guarantee_solvable` restricts training to boards clearable by deduction alone.",
          "Nothing about the agent changes -- same architecture, same three optimization fixes, same seed, same 100,000 episodes.",
          "Because the safe opening also changes what is being measured, the previous chapter's agent was re-scored on this benchmark too, so the two can be compared on identical boards.",
        ],
        headline: "env_v2",
        headlineLabel: "Trained on solvable, first-click-safe boards",
        headlineChange: "Only the board generation differs from the previous chapter",
        status: "published",
        supportingCaption:
          "The training run behind this chapter -- click for its full training curves and hyperparameters:",
        table: {
          caption: "The chapter's weights, scored across every mine density",
          columns: ["Density", "Mines", "Of the board", "Win rate", "95% CI"],
          rows: [
            { label: "Sparse", values: ["8", "9.9%", "97.05%", "96.31 - 97.79"] },
            { label: "Standard", values: ["12", "14.8%", "80.15%", "78.40 - 81.90"], emphasis: true },
            { label: "Dense", values: ["18", "22.2%", "25.45%", "23.54 - 27.36"] },
          ],
          source:
            "2,000 greedy episodes per density, seed 42, on first-click-safe boards with solvability unfiltered. Trained once at standard density and evaluated at the other two without retraining. Not comparable to the previous chapter's table, which is measured on unprotected boards.",
        },
        transferTable: {
          caption: "Zero-shot transfer under this benchmark: the 5x5 weights loaded here untouched",
          columns: ["Density", "Transferred from 5x5", "Trained here", "Δ"],
          rows: [
            { label: "Sparse (8)", values: ["78.25%", "97.05%", "-18.80"] },
            { label: "Standard (12)", values: ["40.10%", "80.15%", "-40.05"], emphasis: true },
            { label: "Dense (18)", values: ["3.30%", "25.45%", "-22.15"] },
          ],
          source:
            "2,000 greedy episodes per cell, seed 42, on first-click-safe boards. `dqn_v2_A_baseline` loaded at 9x9 with no retraining. The transfer column is a generalization result, not a matched one.",
        },
        evaluationsCaption: "Both agents on the protected benchmark, and this one on its own training boards",
        evaluationsSource:
          "2,000 greedy episodes each, seed 42, scoring the weights each run deployed. The first two differ only in the boards they trained on.",
        evaluations: [
          {
            id: "inter-eval-previous",
            label: "Previous chapter's agent, on this benchmark",
            conditions: "Trained on unprotected boards, scored with the first click safe",
            winRate: 0.75,
            ci95: [73.10, 76.90],
            avgEpisodeLength: 22.26,
            losses: 500,
            finding: [
              "75.00%. The carried-config agent, unchanged, simply scored on the protected benchmark.",
              "It scored 52.75% on the boards it actually trained on, so the safe opening alone is worth 22.25 points to it.",
              "That is more than the protected benchmark was worth at 5x5 (16.35 points), which makes sense: a 9x9 game runs about 16 moves, so an unprotected opening has far more games to end prematurely.",
              "It is also the row that makes the comparison below meaningful, by putting both agents on identical boards.",
            ],
          },
          {
            id: "inter-eval-benchmark",
            label: "This chapter's agent, on the same benchmark",
            conditions: "Trained on solvable, first-click-safe boards; scored on the same boards as above",
            winRate: 0.8015,
            ci95: [78.40, 81.90],
            avgEpisodeLength: 21.62,
            losses: 397,
            finding: [
              "80.15% against 75.00%, on identical boards -- a 5.15-point gain (p = 0.0001).",
              "Real, but much smaller than the same change bought at 5x5, where it was worth 22.35 points.",
              "The reason is visible in the row above: at 9x9 most of the benefit is already delivered by the safe opening itself, which the previous agent also receives when scored here.",
              "So the no-guess curriculum adds less as the board grows, because the thing it competes with -- simply not dying on move one -- matters more.",
            ],
          },
          {
            id: "inter-eval-own",
            label: "On its own training distribution",
            conditions: "Solvable boards, first click safe -- where a correct move always exists",
            winRate: 0.9275,
            ci95: [91.61, 93.89],
            avgEpisodeLength: 21.81,
            losses: 145,
            finding: [
              "92.75% where every position has a provably correct move.",
              "Below the 99.65% the same recipe reached at 5x5, and the gap is the point: a 9x9 win needs roughly 22 correct moves in a row against 5, so small per-move error rates compound.",
              "At 145 losses in 2,000 on boards that are always solvable, this agent is making real mistakes rather than being beaten by unavoidable guesses.",
              "That is the opposite of the Beginner result, where 7 losses in 2,000 meant essentially every benchmark loss came from a forced guess.",
            ],
          },
        ],
        limitation: [
          "80.15% on the protected benchmark, against 75.00% for the previous chapter's agent on the same boards -- the no-guess curriculum is worth 5.15 points here.",
          "That is a quarter of what it was worth at 5x5 (22.35 points). The board change still helps, but most of the help now comes from the safe opening rather than the curriculum.",
          "Density remains the hard limit: 97.05% at 8 mines, 80.15% at 12, 25.45% at 18. Adding six mines costs nearly 55 points.",
          "Against the CSP solver on these boards it never pulls ahead: 97.05% against 98.55% at sparse, 80.15% against 89.95% at standard, and 25.45% against 46.85% at dense.",
          "That is the significant difference from Beginner, where the same recipe beat CSP at two densities out of three. The advantage over explicit deduction does not survive the move to a larger board.",
          "Its own training distribution puts the ceiling at 92.75% rather than the 99.65% seen at 5x5, so the agent is now losing games it had the information to win.",
          "Both figures here are single seeds, and only two configurations were run at this board size.",
        ],
        limitationBrief:
          "80.15% and still behind CSP at every density -- the advantage the same recipe held at 5x5 does not survive the larger board.",
      },
    ],
  },
  PPO: {
    beginner: [
      {
        id: "stable-but-flat",
        title: "Stable, and barely above Random",
        about: [
          "PPO tests whether a policy-gradient method does any better here than a value-based one.",
          "Rather than learning what each move is worth and acting greedily, it adjusts the policy directly, with a clipped objective that limits how far one update can move it.",
          "The 25,000-episode run is the reference, matching the budget every DQN arm used.",
          "On top of it we tried the two levers available at this budget: shape the reward for denser feedback, and deploy the best checkpoint rather than the final weights.",
          "Every run on this page uses 10 PPO epochs over each rollout at batch size 32 -- roughly 30,000 gradient updates here. Keeping that fixed across the whole level matters more than it sounds, and the last chapter explains why.",
        ],
        headline: "longer_matched",
        headlineLabel: "25,000 episodes, plain reward",
        headlineChange: "PPO at the project's standard episode budget, matching every DQN arm",
        status: "published",
        supportingCaption:
          "The reference run and the two levers tried on top of it -- click any run for its full training curves and hyperparameters:",
        supporting: [
          { variant: "shaped_matched", label: "25,000 episodes, shaped reward", change: "Cascade bonus and a sharper mine penalty" },
          {
            variant: "shaped_ckpt_matched",
            label: "25,000 episodes, shaped + best checkpoint",
            change: "Shaped reward, plus deploying the best checkpoint instead of the final weights",
          },
        ],
        limitation: [
          "1.05%, against Random's 0.45% on the same boards. It is learning something, but barely.",
          "It trains stably -- nothing diverges, no run destroys its own policy. That is not a point of difference though: the DQN baseline is stable too.",
          "Shaping the reward gives 1.25%. Deploying the best checkpoint gives 1.10%. Both sit inside the noise band, and every confidence interval here overlaps every other.",
          "Checkpoint selection did pick a genuinely different policy and landed on the same win rate, so which weights you deploy is not the problem.",
          "For scale, the CSP solver reaches 43.40% on these same boards, and DQN reaches 11.40% at this episode budget.",
          "So both levers are ruled out, which leaves the budget itself.",
        ],
        limitationBrief:
          "1.05%, and neither reward shaping nor checkpoint selection moves it outside the noise.",
        nextStep:
          "Q-Learning's entire result turned out to be bought by a bigger budget, and DQN's runs were still climbing when they stopped. So the obvious question is whether PPO is simply undertrained -- answerable by giving it four times the episodes.",
      },
      {
        id: "budget",
        title: "Four times the budget changes nothing",
        about: [
          "The same shaped configuration retrained from scratch for 100,000 episodes instead of 25,000.",
          "Nothing else changes -- same epochs, same batch size, same seed -- so the episode count is the only variable, and gradient updates rise with it from about 30,000 to about 123,000.",
          "For the other two learners this was the decisive change: Q-Learning's whole result was bought this way, and DQN's apparent plateau turned out to be a run stopped early.",
          "A discount-factor control runs alongside, since PPO uses gamma 0.99 where DQN uses 0.9, and a near-undiscounted return is much harder for a critic to predict.",
        ],
        headline: "shaped",
        headlineFamily: "ppo_long",
        headlineLabel: "100,000 episodes, shaped reward",
        headlineChange: "Four times the episodes of the previous chapter, nothing else changed",
        status: "published",
        supportingCaption:
          "The longer run and its discount-factor control -- click either for its full training curves and hyperparameters:",
        supporting: [
          {
            variant: "gamma09_matched",
            label: "Tuning control: discount factor 0.9",
            change: "gamma lowered from 0.99 to the 0.9 DQN uses, to test whether the critic's targets were simply too far-sighted",
            family: "ppo_long",
          },
        ],
        limitation: [
          "0.90% against 1.05% at a quarter of the episodes -- a 0.15-point difference in the wrong direction, and not significant (p = 0.63).",
          "That makes PPO the odd one out. The same change was worth about 60% of Q-Learning's total gain, and DQN was still climbing when its run stopped.",
          "Lowering the discount factor changes nothing either: 0.90% either way.",
          "The critic is the obvious suspect. Its explained variance sits at 8% over this run, so the value estimates every policy update depends on are barely better than a constant.",
          "But sitting at 0.90% after four times the training is the more important fact. On these boards more experience simply does not accumulate into a better policy.",
        ],
        limitationBrief:
          "Four times the episodes buys nothing measurable -- unlike every other learner here, the training budget is not what limits PPO on these boards.",
        nextStep:
          "Reward design, checkpoint choice, budget and the discount factor are all now ruled out. That leaves one thing about the agent nobody had varied: every PPO run in this project's history used the same two-conv, Linear-head network, because `--network-size` did not exist for PPO until now.",
      },
      {
        id: "architecture",
        title: "The network, and the control it needed",
        about: [
          "PPO ran on one network for this project's entire history -- two convolutional layers into a flatten-and-Linear head -- not by choice but because no alternative existed for it.",
          "`fully_conv` is the preset DQN's own pipeline ended on: four conv layers, and 1x1 convolutions in place of the Linear head, for both the actor and (via global average pooling) the critic.",
          "It matters most for a reason that has nothing to do with this board. A Linear head is board-size-specific, so a 5x5 PPO model cannot be loaded at 9x9 at all; `fully_conv` is 33,442 parameters at every board size, against 112,122 at 5x5 and 348,722 at 9x9 for the default.",
          "This chapter needed a control run of its own. The obvious comparison -- against the previous chapter's 0.90% -- is not clean: that run predates seed recording and checkpointed on a different interval. So the default network was retrained at the new settings, and the two runs below differ in exactly one key.",
        ],
        headline: "fully_conv",
        headlineFamily: "ppo_long",
        headlineLabel: "Fully-convolutional network",
        headlineChange: "Four conv layers and a 1x1-convolution head -- 33,442 parameters at any board size",
        headlinePosition: "last",
        status: "published",
        supportingCaption:
          "The matched control and the architecture change -- identical but for `network_size`. Click either for its full training curves and hyperparameters:",
        supporting: [
          {
            variant: "shaped_reseeded",
            family: "ppo_long",
            label: "Control: the default network, same settings",
            change: "The previous chapter's configuration re-run at seed 42 and checkpoint_every 5,000, so the architecture is the only difference",
          },
        ],
        limitation: [
          "1.75% against the control's 1.10% -- 0.65 points, and not significant (p = 0.11). The direction is favourable and the evidence does not carry it.",
          "That is almost exactly what the same change was worth to DQN: 0.65 points at p = 0.67. Two different learners, the same architectural change, the same null.",
          "The control run earns its place independently. At 1.10% against the previous chapter's 0.90% for a nominally identical configuration (p = 0.63), it is an unplanned seed replicate -- and a 0.20-point spread from run-to-run luck alone is about a third of the effect this chapter is trying to measure.",
          "Training wins tell a slightly more favourable story than the evaluation does: 1,161 across 100,000 episodes against the control's 889, on the same boards.",
          "The critic got worse, not better. Explained variance ends around 0.03 against the control's 0.07 -- the globally-pooled conv critic predicts less well here than the Linear one it replaced.",
          "Entropy holds higher throughout (1.09-1.34 in the last 40,000 episodes against 0.81-1.01), so the policy stays less committed. On a board where nothing is winning much, that is as easy to read as indecision as exploration.",
          "One limit on this chapter has to be stated plainly: two 3x3 convolutions already give a 5x5 receptive field, which is the entire Beginner board. Both networks here see everything.",
          "So this run tests depth and head shape. It cannot test the receptive-field argument at all, because at 5x5 there is no part of the board either network is missing. That question only exists at 9x9 and above.",
        ],
        limitationBrief:
          "0.65 points and not significant (p = 0.11) -- the same null DQN got. It earns its place by making one model loadable at any board size, not by winning more games.",
        nextStep:
          "So every change to the agent is now null, including its architecture. The one variable never touched is the problem itself: every run so far used the original boards, where the opening click can lose outright.",
      },
      {
        id: "environment",
        title: "The board is what was holding it back",
        about: [
          "The same 100,000-episode shaped configuration, retrained on the two board settings that transformed DQN -- and carrying the previous chapter's `fully_conv` network, so the pipeline ends on the architecture it adopted.",
          "`first_click_safe: area` opens a mine-free 3x3 block on the first move, so no game is lost on move one and every episode begins from a cascade.",
          "`guarantee_solvable` then restricts training to boards clearable by deduction alone, so a correct move always exists.",
          "Both settings were also trained on the default network, and those runs are kept below as an architecture control. That pairing is what makes the board effect measurable without changing two things at once -- and it turned out to matter for one of the findings.",
          "Every run here uses the same gradient budget as the previous chapter, which turned out to matter enormously -- see the last two points below.",
        ],
        headline: "fully_conv",
        headlineFamily: "ppo_v2",
        headlineLabel: "First click safe, fully_conv",
        headlineChange: "The opening click can no longer lose; boards otherwise unfiltered",
        status: "published",
        supportingCaption:
          "The no-guess arm, and the default-network controls both settings were also trained on -- click any run for its full training curves and hyperparameters:",
        supporting: [
          {
            variant: "solvable_fully_conv",
            family: "ppo_v2",
            label: "Safe opening + no-guess boards, fully_conv",
            change: "Training restricted to boards clearable by deduction alone; evaluation stays unfiltered",
          },
          {
            variant: "shaped_matched",
            family: "ppo_v2",
            label: "Architecture control: safe opening, default network",
            change: "The same board setting on the two-conv, Linear-head network -- 7.95% against the headline's 7.90%",
          },
          {
            variant: "solvable_matched",
            family: "ppo_v2",
            label: "Architecture control: no-guess boards, default network",
            change: "The same no-guess setting on the default network -- 4.05% against fully_conv's 6.55%",
          },
        ],
        table: {
          caption: "The chapter's weights, scored across every mine density -- with the default network alongside",
          columns: ["Density", "Mines", "Of the board", "fully_conv", "default net", "Δ"],
          rows: [
            { label: "Sparse", values: ["3", "12.0%", "29.85%", "33.05%", "-3.20 (p = 0.032)"] },
            { label: "Standard", values: ["5", "20.0%", "7.90%", "7.95%", "-0.05 (p = 1.00)"], emphasis: true },
            { label: "Dense", values: ["8", "32.0%", "0.95%", "1.35%", "-0.40 (p = 0.30)"] },
          ],
          source:
            "2,000 greedy episodes per density, seed 42, on first-click-safe boards with solvability unfiltered. Each network trained once at standard density and evaluated at the other two without retraining. Not comparable to the previous chapters' figures, which are measured on unprotected boards.",
        },
        evaluationsCaption: "Both fully_conv policies, scored on solvable boards",
        evaluationsSource:
          "2,000 greedy episodes each, seed 42, scoring the deployed weights. `guarantee_solvable` is training-only, so both are evaluated on unfiltered boards (the headline figures above) as well as on solvable ones here.",
        evaluations: [
          {
            id: "ppo-eval-safe-own",
            label: "Safe-opening policy, on solvable boards",
            conditions: "Trained without the no-guess restriction, scored where a correct move always exists",
            winRate: 0.133,
            ci95: [11.88, 14.86],
            avgEpisodeLength: 3.0,
            losses: 1734,
            finding: [
              "13.30% -- the best number PPO reaches anywhere in this project.",
              "It is measured on boards where a deduction solver wins 100% of the time by construction, and where DQN reaches 99.65%.",
              "So even at its most favourable, and with the compute matched, PPO wins about one game in eight where a value-based agent wins essentially all of them.",
              "The default network scores 12.95% here. A 0.35-point difference at p = 0.78 -- on this benchmark the architecture is simply not the variable.",
            ],
          },
          {
            id: "ppo-eval-noguess-own",
            label: "No-guess policy, on the boards it trained on",
            conditions: "Trained on solvable boards, scored on solvable boards",
            winRate: 0.108,
            ci95: [9.51, 12.24],
            avgEpisodeLength: 3.61,
            losses: 1784,
            finding: [
              "10.80% -- still worse, on its own training distribution, than the policy that never saw a no-guess board (13.30%, a 2.50-point deficit at p = 0.017).",
              "So the curriculum does backfire, and it is not a transfer failure: this is the distribution it was trained for.",
              "But the size of the backfire depends on the architecture, which is the finding this chapter did not expect. On the default network the same comparison is 7.95% against 12.95% -- a 5.00-point deficit. `fully_conv` halves it.",
              "Put the other way round: on no-guess boards the conv head is worth 2.85 points (10.80% against 7.95%, p = 0.0023), where on the safe-opening setting it is worth nothing at all.",
              "The mechanism proposed for this change -- that no-guess boards would make wins frequent enough for an on-policy learner to see them -- is still not supported. It is just less costly than it looked.",
            ],
          },
        ],
        limitation: [
          "7.90% with a safe opening, against 1.75% for the same network on the original boards. That is a 6.15-point gain from changing the board rather than the agent (p < 0.0001), and it is the largest effect anywhere in PPO's pipeline.",
          "So the answer to the previous chapter is that PPO was not undertrained -- it was being asked to learn from a board where roughly a fifth of games were decided before it acted.",
          "The architecture is worth nothing at this density, and the null is about as clean as this project produces: 7.90% against the default network's 7.95%, which is 158 wins against 159 (p = 1.00).",
          "It is not free everywhere, though, and that only became visible once every density was re-measured. At 3 mines the conv head *loses* 3.20 points (29.85% against 33.05%, p = 0.032); at 8 mines it drifts down 0.40 (p = 0.30). So `fully_conv` is neutral where it was tested and mildly negative on sparser boards.",
          "It is still carried forward, because the reason was never win rate: one set of weights that loads at any board size is what the whole next level depends on. But \"architecturally free\" is a claim about Standard density only, and the table above is the reason to say so.",
          "Adding no-guess boards on top still makes it worse -- 6.55% against 7.90% -- though at p = 0.11 that headline difference is not significant on its own. The clearer version of the same result is on solvable boards, where it is 10.80% against 13.30% (p = 0.017).",
          "How much worse depends on the network, which is this chapter's genuine surprise. The default network loses 5.00 points to the curriculum where `fully_conv` loses 2.50. On no-guess boards specifically the conv head is worth 2.85 points (p = 0.0023) -- the only place in PPO's entire pipeline where the architecture does measurable work.",
          "Why is an open question rather than a finding. No-guess boards are a narrower, more structured distribution, and a position-specific Linear head has more room to overfit to one than a translation-equivariant conv head does -- but nothing here tests that, and it should be read as the hypothesis it is.",
          "The discount factor is not neutral either. Gamma 0.9 costs 4.85 points under the safe opening (3.10% against 7.95%), measured on the default network and not re-run under `fully_conv`, so PPO's near-undiscounted return was helping rather than hurting.",
          "Two of those findings are the reverse of what an earlier, under-resourced version of this chapter reported. Those runs used 4 PPO epochs at batch 64 -- about 25,000 gradient updates against 123,000 here.",
          "That is the methodological result worth carrying forward: too little compute did not merely shrink these effects, it inverted two of them. No-guess boards looked like a small gain and are a real loss; the discount factor looked like nothing and costs nearly five points.",
          "For scale, Random scores 1.30% on this benchmark. PPO's best configuration is roughly six times that, and roughly a tenth of the DQN trained on the same boards (77.25%).",
        ],
        limitationBrief:
          "7.90% once the opening is survivable -- 6.15 points up on the same network's original-board result, and still a tenth of DQN on the same benchmark.",
        nextStep:
          "The pipeline now ends on a network whose weights are valid at any board size, which makes one thing testable that never was before: loading this 5x5 policy straight into a 9x9 game with no retraining at all.",
      },
    ],
    intermediate: [
      {
        id: "ppo-intermediate-carried",
        title: "The 5x5 recipe, carried to 9x9",
        about: [
          "The configuration that ended PPO's Beginner pipeline, moved to a 9x9 board with 12 mines and retrained from scratch.",
          "Shaped reward, `fully_conv`, 100,000 episodes, 10 PPO epochs at batch 32, same seed -- nothing re-tuned for the larger board.",
          "This is the same move DQN made between its own Beginner and Intermediate levels, on the same network and the same episode budget, so the comparison between them is now architecture-matched rather than confounded.",
          "The earlier run at this board size used the two-conv, Linear-head design that was the only PPO network available at the time. It is kept below as the architecture control, because the difference between the two turns out to be the interesting part of this chapter.",
          "The epochs-and-batch settings are the matched-compute configuration established at Beginner, carried here unchanged along with everything else. Compute was settled there; this level inherits the answer rather than re-testing it.",
        ],
        headline: "fully_conv",
        headlineLabel: "9x9, 12 mines, carried configuration",
        headlineChange: "The Beginner recipe unchanged -- `fully_conv`, shaped reward -- retrained at the larger board size",
        status: "published",
        supportingCaption:
          "This chapter's run and the earlier default-network one at the same board size -- click either for its full training curves and hyperparameters:",
        supporting: [
          {
            variant: "carried_config",
            label: "Architecture control: the default network",
            change: "The same recipe on two conv layers and a Linear head -- 0 training wins against fully_conv's 4",
          },
        ],
        table: {
          caption: "The same weights, scored across every mine density at this board size",
          columns: ["Density", "Mines", "Of the board", "Win rate", "95% CI"],
          rows: [
            { label: "Sparse", values: ["8", "9.9%", "0.20%", "0.08 - 0.51"] },
            { label: "Standard", values: ["12", "14.8%", "0.00%", "0.00 - 0.19"], emphasis: true },
            { label: "Dense", values: ["18", "22.2%", "0.00%", "0.00 - 0.19"] },
          ],
          source:
            "2,000 greedy episodes per density, seed 42, scoring the deployed weights on unprotected boards. Trained once at standard density and evaluated at the other two without retraining.",
        },
        transferTable: {
          caption: "Zero-shot transfer: the 5x5 weights loaded here untouched, against this chapter's trained model",
          columns: ["Density", "Transferred from 5x5", "Trained here", "Δ"],
          rows: [
            { label: "Sparse (8)", values: ["0.60%", "0.20%", "+0.40"] },
            { label: "Standard (12)", values: ["0.00%", "0.00%", "0.00"], emphasis: true },
            { label: "Dense (18)", values: ["0.00%", "0.00%", "0.00"] },
          ],
          source:
            "2,000 greedy episodes per cell, seed 42, on unprotected boards. `fully_conv` weights are board-size-independent, so `ppo_long_E_fully_conv` loads at 9x9 with no retraining -- a test the old Linear head made impossible. The transfer column is a generalization result, not a matched one.",
        },
        limitation: [
          "0.00% at standard density -- zero wins in 2,000 evaluation episodes. The one non-zero cell anywhere on this page is 0.20% at sparse, four wins in 2,000.",
          "The depth figure is what explains it. A 9x9 board with 12 mines needs 69 correct reveals to win; the median episode ends after 5, and the deepest across 100,000 training episodes reached 36.",
          "So it is not losing winnable games near the end. It never gets close enough to lose one.",
          "The architecture does change the failure mode, even though it does not change the win rate. The default network collapses to a confident non-winning habit -- entropy 2.29 falling to 0.64. This one never collapses at all, holding 2.2-2.9 for the whole run.",
          "It also plays deeper (5.4 moves per episode against 4.8, deepest 36 against 29) and records the first training wins this board size has produced: 4 in 100,000, against 0 for the default network.",
          "That is a real difference and it is nowhere near enough. Four wins in a hundred thousand games is a policy that has stopped converging on the wrong answer without starting to find the right one.",
          "Its value head is weakly predictive either way: explained variance holds around 0.27, against 0.55 for the safe-opening run in the next chapter.",
          "Compute is not available as an explanation. This run inherits Beginner's matched configuration and lands at roughly 150,000 gradient updates against DQN's 343,000 -- the same order, on a question that was already settled one level down.",
          "For scale, DQN's identical carried-config move to 9x9 scores 52.75%, and PPO's own Beginner version of this recipe scores 1.75%.",
        ],
        limitationBrief:
          "0.00% at standard density, 0.20% at sparse. The conv head stops the policy collapsing and buys 4 training wins in 100,000 -- and no measurable win rate.",
        nextStep:
          "At Beginner the board settings were the only change that ever moved PPO, taking it from 1.75% to 7.90% where every change to the agent had been null. That makes them the obvious next test here -- and on a 9x9 board, where a game runs far longer, an unprotected opening has many more episodes to end prematurely.",
      },
      {
        id: "ppo-intermediate-environment",
        title: "The board change that worked everywhere else",
        about: [
          "The identical configuration retrained with `first_click_safe: area`, which keeps the 3x3 block around the opening click mine-free.",
          "Nothing about the agent changes -- same `fully_conv` network, same shaped reward, same seed, same 100,000 episodes, same gradient budget.",
          "This is the single change that produced the largest gain in PPO's Beginner pipeline (1.75% to 7.90%) and DQN's largest at this board size (75.00% to 80.15% like-for-like).",
          "The earlier default-network run at this setting is kept below as the architecture control, the same pairing the previous chapter uses.",
          "Because the safe opening also changes what is being measured, the previous chapter's agent was re-scored on this benchmark too, so the two can be compared on identical boards.",
        ],
        headline: "env_v2_fully_conv",
        headlineLabel: "Trained with the first click safe, fully_conv",
        headlineChange: "Only the board generation differs from the previous chapter",
        status: "published",
        supportingCaption:
          "This chapter's run and the earlier default-network one at the same setting -- click either for its full training curves and hyperparameters:",
        supporting: [
          {
            variant: "env_v2",
            label: "Architecture control: the default network",
            change: "The same safe-opening setting on two conv layers and a Linear head -- 1 training win against fully_conv's 7",
          },
        ],
        table: {
          caption: "The chapter's weights, scored across every mine density",
          columns: ["Density", "Mines", "Of the board", "Win rate", "95% CI"],
          rows: [
            { label: "Sparse", values: ["8", "9.9%", "0.70%", "0.42 - 1.17"] },
            { label: "Standard", values: ["12", "14.8%", "0.00%", "0.00 - 0.19"], emphasis: true },
            { label: "Dense", values: ["18", "22.2%", "0.00%", "0.00 - 0.19"] },
          ],
          source:
            "2,000 greedy episodes per density, seed 42, on first-click-safe boards with solvability unfiltered. Trained once at standard density and evaluated at the other two without retraining. Not comparable to the previous chapter's table, which is measured on unprotected boards.",
        },
        transferTable: {
          caption: "Zero-shot transfer under this benchmark: the 5x5 weights loaded here untouched",
          columns: ["Density", "Transferred from 5x5", "Trained here", "Δ"],
          rows: [
            { label: "Sparse (8)", values: ["1.05%", "0.70%", "+0.35"] },
            { label: "Standard (12)", values: ["0.05%", "0.00%", "+0.05"], emphasis: true },
            { label: "Dense (18)", values: ["0.00%", "0.00%", "0.00"] },
          ],
          source:
            "2,000 greedy episodes per cell, seed 42, on first-click-safe boards. `ppo_v2_J_fully_conv` loaded at 9x9 with no retraining. The transfer column is a generalization result, not a matched one.",
        },
        evaluationsCaption: "Both agents on the protected benchmark, scored on identical boards",
        evaluationsSource:
          "2,000 greedy episodes each, seed 42, scoring the weights each run deployed. The two differ only in the boards they trained on.",
        evaluations: [
          {
            id: "ppo-inter-eval-previous",
            label: "Previous chapter's agent, on this benchmark",
            conditions: "Trained on unprotected boards, scored with the first click safe",
            winRate: 0,
            ci95: [0.0, 0.19],
            avgEpisodeLength: 6.04,
            losses: 2000,
            finding: [
              "0.00%. The previous chapter's `fully_conv` agent, unchanged, simply scored on the protected benchmark.",
              "The equivalent row for DQN gains 22.25 points from this alone, and for PPO at 5x5 the protected benchmark is worth several points on its own.",
              "Here it is worth nothing measurable. The safe opening does lengthen its episodes -- 6.04 moves against 5.56 on unprotected boards -- and converts none of that into a win.",
              "That is the first sign that whatever limits this agent at 9x9 is not located at the start of the episode.",
            ],
          },
          {
            id: "ppo-inter-eval-benchmark",
            label: "This chapter's agent, on the same benchmark",
            conditions: "Trained with the first click safe; scored on the same boards as above",
            winRate: 0,
            ci95: [0.0, 0.19],
            avgEpisodeLength: 5.72,
            losses: 2000,
            finding: [
              "0.00% against 0.00%, on identical boards. Training under the safe opening is worth nothing measurable at this board size.",
              "It is not that the change failed to apply. It did exactly what it was designed to do -- no episode is lost on move one any more.",
              "Removing that entire failure mode moved the win rate not at all, which is a cleaner result than a small gain would have been. The opening was simply never the binding constraint here.",
              "The safe opening plus the conv head do buy depth: episodes run 5.72 moves against the default network's 4.90, and 139 episodes per 100,000 reach 20+ moves against 29.",
              "A win needs 69. Doubling the number of episodes that reach 20 moves is real and it is still an order of magnitude short of the target.",
              "7 wins appeared across 100,000 training episodes, against 1 for the default network at the same setting. That is the entire measured difference, and it does not survive into evaluation.",
            ],
          },
        ],
        limitation: [
          "0.00% at standard density on the protected benchmark, against 0.00% for the previous chapter's agent on the same boards. The change that mattered most everywhere else is worth nothing here.",
          "The mechanism is visible rather than inferred. The safe opening removes an entire failure mode -- no episode can be lost on move one -- and the win rate does not move.",
          "The one non-zero cell anywhere at this board size is this agent at sparse density: 0.70%, fourteen wins in 2,000. That is double the default network's 0.35%, and at p = 0.19 it is not a difference this evidence can carry.",
          "Its value head is the best PPO produces at 9x9 (explained variance 0.51 against 0.27 without the safe opening), and its entropy never collapses. It converted neither into wins.",
          "Depth is where the architecture shows up. Episodes reaching 20+ moves run 139 per 100,000 against the default network's 29 -- a real gain, against a target of 69 reveals.",
          "Against DQN on the same boards and the same budget: 80.15% to this run's 0.00%. Against CSP: 89.95%.",
          "All four runs at this board size are single seeds, and no configuration was replicated.",
        ],
        limitationBrief:
          "0.00% against 0.00% on identical boards. Removing an entire failure mode moved the win rate not at all.",
      },
    ],
  },
};

/** This level's chapters, or `undefined` when it should render one card per
 * run the old way. */
export function chaptersFor(agentName: string, level: PipelineLevel): PipelineChapter[] | undefined {
  return PIPELINE_CHAPTERS[agentName]?.[level];
}

/** Families these chapters reference beyond the level's own, so `LevelStory`
 * knows what else to fetch. Empty for a level whose chapters all draw from
 * one family. */
export function extraFamiliesFor(chapters: PipelineChapter[] | undefined, ownFamily: string): string[] {
  if (!chapters) return [];
  const families = new Set<string>();
  for (const chapter of chapters) {
    if (chapter.headlineFamily) families.add(chapter.headlineFamily);
    for (const support of chapter.supporting ?? []) {
      if (support.family) families.add(support.family);
    }
  }
  families.delete(ownFamily);
  return [...families];
}

/** How a run is keyed across families -- variant labels are not unique on
 * their own (both PPO families have a `baseline`). */
export function runKey(family: string, variant: string): string {
  return `${family}:${variant}`;
}
