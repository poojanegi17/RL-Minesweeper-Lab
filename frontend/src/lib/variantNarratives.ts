import type { PipelineLevel } from "@/lib/levelPipelines";

/**
 * Hand-authored, real-data-grounded per (agent, level, variant) narrative.
 * Beginner's entries paraphrase the project README's DQN/PPO ablation tables
 * ("### DQN" / "### PPO" sections) -- the same source `ResearchPipeline`'s
 * `STEPS` and `BoardConfigComparisonTable`'s `TREND_NOTES` already draw from.
 * Intermediate/Expert's entries come from this project's own later
 * generalization/exploration runs (real per-episode training curves,
 * inspected directly -- see each entry for the specific evidence). Keyed by
 * the real `RunBrief.variant` string the backend returns (`baseline`,
 * `checkpoint`, `lr_decay`, `high_entropy`, ...), not invented labels. A
 * variant with no entry here just renders without the extra "about"/
 * "limitation" text (see `ExperimentSetup`'s fallback).
 */
export interface VariantNarrative {
  /** One line: what this variant is / why it was tried. */
  about: string;
  /** What this variant taught us and what was still wrong with it, as short
   * plain-language points -- rendered both as the card's "What did we learn?"
   * list and as the connector's "Limitation" box leading into the next variant
   * (or, on the last variant so far, beside that level's closing
   * `VARIANT_LEVEL_CONCLUSIONS` entry). Authored as points rather than a
   * paragraph because these findings have several independent parts -- a
   * number, a mechanism, a consequence -- that read badly welded together. */
  limitation: string[];
  /** One short line for the connector between this card and the next -- the
   * headline shortcoming only. The connector is a signpost, not a report, so
   * the full `limitation` list stays inside the card. Falls back to the first
   * `limitation` point when absent. */
  limitationBrief?: string;
  /** What was tried next and which shortcoming above it targeted. Rendered in
   * the same connector, so the flow from one card to the next states its own
   * reasoning instead of leaving the reader to infer it. Omitted on the last
   * card at a level, where `VARIANT_LEVEL_CONCLUSIONS` closes the story. */
  nextStep?: string;
}

export const VARIANT_NARRATIVES: Record<string, Partial<Record<PipelineLevel, Record<string, VariantNarrative>>>> = {
  DQN: {
    beginner: {
      baseline: {
        about: "Double DQN over the 11-channel encoding, with experience replay and a target network, and no stability fixes layered on top -- 25,000 episodes at lr 1e-4, a constant learning rate, the default network, and the final weights deployed as-is. Everything above is already justified; nothing here is tuned. This is the floor every other variant is measured against.",
        limitation: [
          "0.55% win rate [0.31-0.98] -- barely above Random's 0.45% under the same v1 environment, so as a deployed policy this is close to worthless.",
          "It does learn: the training win rate climbs to 3.28% by around episode 7,500.",
          "Then it decays for the remaining 17,500 episodes. It learns a decent policy and destroys it.",
          "The cause is visible in the value estimates. Average max Q-value rises from 18 to 290,825 -- against a ceiling of 10, which is the most this reward can possibly be worth.",
          "A Q-value above 10 isn't just high, it's impossible. That points at a broken update rule rather than a tuning problem.",
        ],
        limitationBrief:
          "Learns a decent policy by episode 7,500, then destroys it -- and reports Q-values of 290,825 where 10 is the most the reward can be worth.",
        nextStep:
          "A value that high isn't merely wrong, it's impossible, which points at the update rule rather than any hyperparameter. Of the three terms in the bootstrap target, only one is unbounded: the next-state Q-value. So we audited that -- and found its argmax was ranging over cells the agent can never click.",
      },
      masked_shaped: {
        about: "Swaps the training reward for the environment's 'shaped' mode: still +1 per safe reveal, plus 0.2 for every extra cell a cascade opens, with a sharper -15 for a mine and +20 for a win. Evaluation still uses the plain reward, so the win rate stays comparable. Everything else matches the masked-target run exactly, making the reward the only variable.",
        limitation: [
          "11.80% against 11.40% for the plain reward -- no measurable difference (p = 0.69).",
          "Its training curve actually ran slightly *below* the plain-reward run at almost every stage, finishing on 1,272 training wins against 1,360.",
          "So the denser signal added nothing. A reasonable hypothesis, cleanly tested, and wrong.",
          "In hindsight it fits the calibration finding: a big cascade already raises the value of the resulting state, because fewer clicks remain to win. The bonus was telling the agent something its value function had already worked out.",
          "Useful conclusion by elimination -- with the value estimates correct and a denser reward changing nothing, the learning signal is not what limits this agent.",
        ],
        limitationBrief:
          "A denser reward changed nothing (11.80% vs 11.40%, p = 0.69) -- the learning signal isn't the bottleneck.",
        nextStep:
          "If the signal isn't the limit, the data might be. The agent reaches its 5% exploration floor by episode 598 of 25,000, so for 98% of training it plays nearly greedily and keeps revisiting the same narrow set of boards. Stretching that descent across 15,000 episodes tests whether broader coverage is what it was missing.",
      },
      masked_slow_epsilon: {
        about: "Stretches the exploration schedule: epsilon decays at 0.9998 per episode instead of 0.995, so it reaches its 0.05 floor around episode 14,977 rather than 598. The agent therefore spends most of training genuinely exploring rather than nearly greedy. Nothing else differs from the masked-target run.",
        limitation: [
          "12.30% against 11.40% for the short schedule -- again no measurable difference (p = 0.38).",
          "The two effects predicted beforehand appear to have cancelled. Broader coverage helped a little: the win rate was still rising at the end (9.80% in the final block) where the short-schedule run had flattened and dipped.",
          "But random moves hit mines, so episodes ran consistently shorter (4.9 clicks against 5.2), producing shallower data -- the opposite of what the depth problem needs.",
          "With this tested, all four post-fix runs form a single statistical cluster: 11.40%, 11.80%, 12.30%, 12.85%, with none of the six pairwise comparisons significant.",
          "So DQN has a real plateau at roughly 12% here, and neither the learning rate, the reward, nor the exploration schedule moves it. CSP reaches 43.40% on the same board under the same v1 environment.",
        ],
        limitationBrief:
          "Also neutral (12.30% vs 11.40%, p = 0.38). All four post-fix runs now sit in one cluster around 12% -- a real plateau.",
        nextStep:
          "Three levers ruled out, so the next step came from measuring *where* the losses are rather than guessing. Replaying the best model against the CSP solver move by move: 30% of its games are lost while a provably safe cell was sitting on the board, and when such a cell exists it picks one only about two-thirds of the time. That is a deduction failure, not a guessing failure. Since deduction is iterative -- a revealed 0 resolves in one step, the subset rule needs two constraints compared, a chained deduction needs three -- and a two-layer convolutional stack can only perform about two rounds of it, network depth became the obvious candidate.",
      },
      masked_deep: {
        about: "Four convolutional layers instead of two, at the same widths, so depth is the only variable (+17% parameters). The hypothesis was specific: Minesweeper deduction is iterative, and a fixed two-layer stack can only perform about two rounds of it, where the CSP solver iterates its rules until nothing new can be deduced.",
        limitation: [
          "6.35% -- decisively worse than the two-layer network's 11.40%, not better.",
          "And it failed on exactly the measure that motivated it: it picks a provably safe cell 48.4% of the time against the shallower network's 66.6%, with avoidable losses rising from 25.7% to 40.7%.",
          "So the hypothesis is refuted on its own terms. Depth did not buy better deduction.",
          "One number complicates the reading, though. Its average max Q-value only reached 4.7, where the shallower network converges to 10.1 -- and 10 is the correct value under this reward. The deep network never finished learning.",
          "So this run establishes that depth is worse at a matched 25,000-episode budget. It cannot distinguish 'depth does not help' from 'depth needs more episodes than we gave it'. A longer deep run would separate those.",
        ],
        limitationBrief:
          "Depth made things worse, not better (6.35% vs 11.40%) -- and it deduces less well too, 48.4% against 66.6%.",
        nextStep:
          "Its value function never converged (max Q 4.7 against the correct 10), so the immediate question is whether the deeper network simply needs a longer budget. Separately, the shallower network reaches the correct value scale and still only deduces two-thirds of the time -- meaning it has learned the right magnitudes without the right per-action ranking, which is a different problem from anything tested so far.",
      },
      checkpoint: {
        about: "Deploys whichever checkpoint scored the best win rate during training, instead of just whatever's left in memory when it ends. Same training run as the baseline -- only the deployed weights differ.",
        limitation: [
          "2.10% -- roughly 4x the same run's final weights (p < 0.0001), the biggest single jump in the original ablation.",
          "It works only because the run destroys itself. Saving a snapshot near the episode-7,500 peak rescues a policy that later training wrecks.",
          "It fixes nothing about training. Identical run, identical collapse -- just a better moment to stop reading from.",
          "Which checkpoint gets picked is itself noisy: candidates are ranked on only 50 episodes each.",
        ],
        limitationBrief:
          "Rescues a collapsing run rather than improving one -- and stops being needed once the collapse is gone.",
        nextStep:
          "Next we attacked the collapse itself instead of working around it, by decaying the learning rate to slow the growth in Q-values.",
      },
      lr_decay: {
        about: "Lowers the learning rate on a schedule over training, targeting the late-training divergence directly. Identical to the baseline until episode 10,000, where the first milestone kicks in.",
        limitation: [
          "1.60% from the final weights -- a real gain over the baseline's 0.55% (p = 0.0013).",
          "Divergence slows by about 300x: final max Q-value around 900 instead of 290,000.",
          "But it doesn't stop it. 900 is still about 90x the ceiling of 10, so the run is still heading the wrong way, just slower.",
          "Combined with best-checkpoint deployment it adds nothing measurable (2.40% vs 2.10%, p = 0.52) -- both are working around the same underlying failure.",
        ],
        limitationBrief:
          "Slows divergence roughly 300x but never stops it: Q-values still end ~90x above what the reward can justify.",
        nextStep:
          "So we questioned the premise instead: perhaps a network with less capacity would be inherently harder to push into runaway estimates.",
      },
      small_net: {
        about: "An alternate hypothesis: maybe the real problem is model capacity vs. stability, not the learning-rate schedule.",
        limitation: [
          "2.20% from its best checkpoint -- indistinguishable from every other run once the deployed weights are chosen the same way.",
          "So capacity had no detectable effect in either direction.",
          "It diverged too, peaking at max Q-value 9,961. A smaller network slowed the runaway; it didn't prevent it.",
          "That's informative: capacity controls what the network *can* represent, not whether the target it's fitting makes sense.",
        ],
        limitationBrief:
          "Capacity had no detectable effect -- the smaller network diverged too, just more slowly.",
        nextStep:
          "With two partial fixes and no cause identified, the last thing left to try was combining them, to see whether their benefits stacked.",
      },
      masked_lr_decay: {
        about: "The control the earlier ablation never had: the masked bootstrap plus a decaying learning rate (1e-4 -> 5e-5 -> 1e-5), the same schedule a pre-fix variant used. If LR decay was a real fix rather than a brake on a runaway feedback loop, it should still help here.",
        limitation: [
          "12.85% from the final weights against 11.40% for the mask alone -- not a significant difference (p = 0.16).",
          "So LR decay is neutral once the bug is fixed, versus the roughly 3x effect it appeared to have before.",
          "It does still cap the Q-values, which plateau at 5.2 instead of climbing to 10.1 -- exactly what a brake on a runaway looks like.",
          "Best-checkpoint selection now actively hurts: final weights 12.85% vs the chosen 'best' checkpoint at 7.20% (p = 3e-9).",
          "Ranking candidates on 50-episode evaluations picks a model nearly 2x worse than simply taking the end of training.",
        ],
        limitationBrief:
          "Adds nothing measurable over the fix alone (12.85% vs 11.40%, p = 0.16) -- it was only ever slowing the runaway.",
        nextStep:
          "That settles the stability question. The agent's value estimates are now correct, but it still dies after about 5 clicks when a win needs 11 -- so the remaining gap is skill, not stability. The most obvious mis-specification left is the reward: a click that cascades open 8 cells earns the same +1 as one revealing a single cell, giving no signal that separates a high-information move from a marginal one. So we tried a denser reward.",
      },
      masked_target: {
        about: "Not a hyperparameter at all -- a correctness fix. The Double DQN bootstrap took its argmax over every cell, including ones already revealed in the next state. Those actions can never be chosen by `select_action`, so they receive no gradient and their Q-values drift freely; the unmasked argmax then fed that drift back in as a target. Restricting the argmax to hidden cells is the whole change, and nothing else about the baseline configuration was touched.",
        limitation: [
          "11.40% from the final weights -- a 21x jump over the identical configuration without the mask (p ~ 2e-47). The largest effect anywhere in this project.",
          "The divergence is gone, not reduced: average max Q-value settles at 10.1, against a ceiling of 10.0.",
          "Peak loss falls from 70,741 to 5.5.",
          "The training curve changes shape. Instead of peaking at episode 7,500 and decaying, the win rate climbs the whole way: 1.0% -> 3.6% -> 6.6% -> 8.8% -> 10.1%.",
          "Two earlier conclusions no longer hold. Final weights now beat the best checkpoint, so checkpoint selection is unnecessary -- and training longer is no longer harmful.",
          "This is headroom recovered at 5x5 only. The fix was never run at a larger board on its own, and the recipe that eventually carried to 9x9 came several chapters later.",
        ],
        limitationBrief:
          "Fixes the divergence completely and lifts the win rate 21x -- but only at 5x5. Bigger boards are untouched.",
        nextStep:
          "The classic remedy for unstable Q-learning is a decaying learning rate, and it appeared to help by 3x before the fix. Now that the runaway is gone, we re-ran it as a control to find out whether it was ever doing real work.",
      },
      combined: {
        about: "LR decay + checkpointing together -- do the two independently-validated fixes compound? Nominally identical in configuration to the LR-decay variant, so it doubles as an accidental replicate.",
        limitation: [
          "2.10% -- no better than best-checkpoint deployment on its own.",
          "That's expected rather than puzzling: both fixes address the same collapse, so using both can't beat the better one.",
          "Its real value was accidental. Its settings are identical to the LR-decay run, making it an unplanned repeat of that experiment.",
          "The two scored 1.25% and 1.60% on final weights -- a ~0.35pp spread from run-to-run luck alone, which is about as large as the effects being measured.",
          "So every conclusion at this level should be treated as provisional until each variant is repeated across several seeds.",
        ],
        limitationBrief:
          "The two fixes don't compound, because both were treating the same underlying failure.",
      },
    },
    intermediate: {
      baseline: {
        about: "Double DQN with the winning 5x5 fix (LR decay) applied unchanged, on a 9x9 board -- the starting point for Intermediate's own exploration.",
        limitation: [
          "Zero wins in 2,000 evaluation episodes (95% upper bound 0.19%).",
          "Reward looked like it just needed more training -- it drifted slowly upward, -5.96 to -5.54.",
          "The value estimates said otherwise: average loss quadrupled (6.9 -> 29.0) and average max Q-value rose six-fold (37 -> 218).",
          "218 against a ceiling of 10 is the same runaway seen at 5x5 -- so the winning 5x5 recipe did not transfer.",
          "Only 1 win across all 25,000 training episodes.",
        ],
      },
      lower_lr: {
        about:
          "Halves the entire learning-rate schedule (5e-5 -> 2.5e-5 -> 5e-6, down from the winning 5x5 recipe's 1e-4 -> 5e-5 -> 1e-5) -- everything else (target-sync frequency, network size, batch size) unchanged, directly targeting the loss/Q-value divergence the baseline showed here.",
        limitation: [
          "Still effectively zero: 1 win in 2,000 (upper bound 0.28%).",
          "Stability improved a lot -- average loss ended at 6.9 instead of 29.0, and max Q-value at 46 instead of 218.",
          "But both were still climbing at the end, so this is progress rather than a fix.",
          "A checkpoint flagged 'best' after 1 win in a 50-episode check won nothing over 2,000. That flag was noise.",
        ],
      },
      lower_lr_further: {
        about:
          "Halves the learning rate again from the previous variant (2.5e-5 -> 1.25e-5 -> 2.5e-6, a quarter of the original 5x5 recipe) -- continues the validated direction to see how far the stability keeps improving.",
        limitation: [
          "Still effectively zero: 1 win in 2,000 (upper bound 0.28%).",
          "Training is now genuinely stable -- loss flat around 4, max Q-value 16.8, essentially unchanged across the whole run.",
          "That makes it the only DQN run in the project whose Q-values stayed near the ceiling of 10.",
          "And it changed nothing: reward barely moved and the win rate stayed at zero.",
          "Because stability was never the blocker. A win at 9x9 takes about 26 clicks; this run's deepest episode out of 25,000 was 30, and only 0.03% of transitions ever got that deep.",
          "The agent almost never saw a win, so there was nothing to learn from -- no matter how well-behaved its Q-values became.",
        ],
      },
    },
  },
  PPO: {
    beginner: {
      baseline: {
        about: "6,000 episodes, the default (non-shaped) reward signal -- the starting point every other variant is measured against.",
        limitation: [
          "0.75% win rate -- the starting point for this level.",
          "The open question was whether more training or a different reward signal would move it.",
          "The two variants that follow answer it: nothing measurably did.",
        ],
      },
      longer: {
        about: "Same default reward, 25,000 episodes instead of 6,000 -- tests whether the baseline just needed more training time.",
        limitation: [
          "0.95% -- indistinguishable from the baseline's 0.75% (p = 0.49).",
          "If anything it is slightly better, not worse.",
          "So 'just train longer' is neither confirmed nor ruled out here. The experiment could not resolve a difference this small.",
        ],
      },
      shaped: {
        about: "A denser reward signal for larger cascade reveals, instead of just win/lose -- targets PPO's sparse-reward credit-assignment problem directly.",
        limitation: [
          "1.05% -- the best PPO configuration, but not significantly better than the baseline (p = 0.32).",
          "The real support comes from the critic rather than the win rate: explained variance 10.2%, the best of the three runs.",
          "That is measured across hundreds of updates instead of a couple of hundred games, so it is the more trustworthy signal.",
          "Read shaping as directionally right and mechanically motivated -- not as a demonstrated win-rate gain.",
        ],
      },
      shaped_checkpoint: {
        about: "Reward shaping plus best-checkpoint deployment -- the same safety net DQN's own ablation validated, applied here too. Shares its training run with the shaped variant; only the deployed weights differ.",
        limitation: [
          "1.05% -- the same score as the final weights.",
          "Not because they are the same weights: the checkpoint is from episode 15,000, the final from 25,000. They simply won the same number of games.",
          "Unlike DQN, PPO shows no peak-then-collapse for checkpointing to rescue.",
          "That is consistent with PPO never diverging the way DQN's Q-values did.",
        ],
      },
    },
    intermediate: {
      baseline: {
        about: "The winning 5x5 fix (reward shaping), unchanged, on a 9x9 board -- the starting point for Intermediate's own exploration.",
        limitation: [
          "Zero wins in 2,000 evaluation episodes.",
          "Reward plateaued about a quarter of the way in and stayed flat for the remaining 19,000 episodes.",
          "Entropy collapsed throughout (2.61 -> 1.05): the policy settled into one confident, non-winning habit.",
          "Flat reward plus shrinking entropy is the signature of a local optimum.",
          "Only 1 win across all 25,000 training episodes.",
        ],
      },
      high_entropy: {
        about: "Same shaped-reward setup, entropy bonus raised 5x (0.01 -> 0.05) -- directly targets the premature entropy collapse the baseline showed here.",
        limitation: [
          "Still zero wins in 2,000 (upper bound 0.19%).",
          "A checkpoint scored 2% on a 50-episode check and was flagged 'best'; over 2,000 episodes it won nothing.",
          "With about 50 candidates each ranked on 50 episodes, a false flag like that is close to inevitable.",
          "Raising the entropy bonus on its own does not make 9x9 winnable.",
        ],
      },
      low_lr_high_entropy: {
        about:
          "High entropy (0.05) combined with a lower learning rate (1e-4, vs. 3e-4) -- targets what the high-entropy variant's own curve suggested: entropy still collapsed even with the bonus raised, meaning policy updates were sharpening the distribution faster than the entropy bonus could counteract. Slowing the learning rate down should let exploration actually persist.",
        limitation: [
          "Zero wins in 2,000, and zero across all 25,000 training episodes -- fewer than the baseline's 1.",
          "The mechanism worked exactly as intended: entropy held steady at 3.4-3.6 instead of collapsing early.",
          "And the reward still never improved (-6.5 to -6.7 throughout).",
          "So entropy collapse was a symptom, not the cause. Fixing it changed nothing.",
          "One caveat: approx_kl fell to 0.003 and clip fraction to 0.025, meaning the policy barely updated at all -- so the stable entropy is partly just a frozen policy.",
          "PPO also got about 7,400 gradient steps here against DQN's 135,000 for the same nominal 25,000 episodes. These runs are under-trained more than under-tuned.",
        ],
      },
    },
  },
};

/** Each level's closing synthesis so far -- shown once, after the last
 * variant card explored at that level, alongside its own limitation instead
 * of motivating a "next variant" that doesn't exist yet. Beginner's entries
 * are verbatim from the README's own "Takeaway"/"Honest finding" for that
 * agent; Intermediate/Expert's are this project's own real generalization
 * findings, written as an honest status (some still actively being explored,
 * not a closed investigation) rather than a final verdict. */
export const VARIANT_LEVEL_CONCLUSIONS: Record<string, Partial<Record<PipelineLevel, string[]>>> = {
  DQN: {
    beginner: [
      "On the original boards this level went from 11.40% to 38.55%. On first-click-safe boards the same architecture reaches 77.25%. Those are two different benchmarks and should never be subtracted from one another.",
      "Almost none of the gain came from the hyperparameters anyone would reach for first.",
      "The four standard levers -- learning-rate decay, reward shaping, a longer exploration schedule, a deeper network -- were tried first and moved nothing. Every arm landed between 6.35% and 12.85%.",
      "They looked like a ceiling. They were not. The same baseline trained four times longer reaches 21.25%, so all four had been sweeping hyperparameters against a run that had not finished learning. That is the methodological lesson of this level: measure whether you have converged before you tune.",
      "Three settings nobody had varied then took it to 37.90% -- putting TD targets inside the Huber loss's quadratic regime, stretching exploration, and cutting the replay ratio. They are worth 16.65 points together and cannot be separated: applied without the replay-ratio change, the other two score 5.95%, far worse than doing nothing at all.",
      "The architecture change was a null. Replacing the board-size-specific Linear head with 1x1 convolutions is worth 0.65 points (p = 0.67). It earns its place by making one model loadable at any board size, not by winning more games.",
      "The largest single gain came from changing the board rather than the agent. Trained on no-guess, first-click-safe layouts, the same architecture reaches 77.25% where the version trained on unprotected boards manages 54.90% on those same boards -- 22.35 points, confidence intervals disjoint.",
      "Two caveats sit on that headline. The protected benchmark is an easier game in its own right, worth 16.35 points to an agent that never trained on it; and the two board settings moved together, so no-guess boards and the safe opening are not yet separated.",
      "Everything here is a single seed, and no configuration was replicated -- so any difference this page calls significant rests on one run per arm.",
      "One measured caveat on precision: the 100,000-episode baseline scores 21.25% on its own 2,000 evaluation boards and 19.15% on a different 2,000, so the evaluation set alone moves a figure by around two points.",
      "Five earlier runs are excluded from this page entirely. They predate a correctness fix to the bootstrap target -- its argmax ranged over already-revealed cells, whose Q-values receive no gradient and drift freely -- and so measure an implementation that no longer exists. The README keeps the full record.",
    ],
    intermediate: [
      "The Beginner recipe transfers without retuning. On the original boards it reaches 52.75% at 9x9 against 38.55% at 5x5, and on the protected benchmark 80.15% against 77.25%.",
      "The larger board is easier for it, for two reasons: mine density is lower (14.8% against 20.0%) and a bigger grid leaves more of the board constrained by revealed numbers at any moment.",
      "It is harder in one respect that matters later -- episodes run about 22 moves here against 5, so a win needs correct play sustained four times as long.",
      "This level only exists because of the conv head. The old Linear head was board-size-specific, so a 5x5 model could not be built at 9x9 at all. The change that bought no accuracy at Beginner is what made the transfer possible, which is the clearest argument for keeping it.",
      "The board changes still help, but much less than at 5x5. Training on no-guess boards is worth 5.15 points here against 22.35 at Beginner, because at 9x9 most of the benefit already comes from the safe opening itself -- worth 22.25 points to an agent that never trained with one.",
      "The headline finding is a loss rather than a gain. At 5x5 this recipe beat the CSP solver at two densities out of three; at 9x9 it trails CSP at all three (97.05% against 98.55%, 80.15% against 89.95%, 25.45% against 46.85%).",
      "Its own training distribution shows why. Given boards where a correct move always exists it reaches 92.75%, not the 99.65% seen at 5x5 -- so it is now losing games it had the information to win, and the deficit to deduction is the agent's own error rate rather than bad luck.",
      "Density is still the hard limit, and it bites harder than board size: 97.05% at 8 mines falls to 25.45% at 18.",
      "Three earlier attempts at this board scored 0.00%, 0.05% and 0.05%. They predate every change in the Beginner pipeline and are excluded rather than compared against.",
      "Only two configurations were run at this board size, both single-seed, and neither was tuned for 9x9 -- so this level establishes that the recipe carries, not that it is the right recipe here.",
    ],
  },
  PPO: {
    beginner: [
      "PPO ends this level at 7.90%, against Random's 1.30% and DQN's 77.25% on the same boards. It learns, and it is not competitive.",
      "Everything done to the agent was null. Reward shaping, checkpoint selection, four times the training budget and finally the network architecture all left it inside the noise band, at 0.80% to 1.75% on the original boards.",
      "The architecture was the last of those to be tested, and it is null twice over. On the original boards a fully-convolutional network is worth 0.65 points against a matched control (1.75% against 1.10%, p = 0.11); once the opening is made safe it is worth nothing at all -- 7.90% against 7.95%, 158 wins against 159, p = 1.00.",
      "It is carried forward anyway, on the same terms DQN carried it: 33,442 parameters at every board size, so one set of weights loads at 9x9 or 16x16. That is a transfer argument and it should never be presented as a win-rate one.",
      "That test has a boundary worth stating: two 3x3 convolutions already cover the whole 5x5 board, so this level can measure depth and head shape but is structurally unable to say anything about receptive field. That question only exists at larger boards.",
      "What did move it was the board. Giving the opening click a guaranteed safe 3x3 block took the same network from 1.75% to 7.90% -- a 6.15-point gain, and the only large effect in PPO's pipeline.",
      "So the honest diagnosis is not that PPO was undertrained. It was being asked to learn from a distribution where roughly a fifth of games ended before it had made a real decision, and no amount of extra experience fixes that.",
      "Two things expected to help made it worse. Restricting training to no-guess boards is a real loss -- 10.80% against 13.30% on solvable boards, where the policy that never saw them does better on their own distribution (p = 0.017). Lowering the discount factor to DQN's 0.9 cost a further 4.85 points.",
      "The no-guess result comes with a qualification this level did not anticipate, and it is the one place where the architecture does measurable work. The default network loses 5.00 points to that curriculum; `fully_conv` loses 2.50. On no-guess boards specifically the conv head is worth 2.85 points (p = 0.0023), having been worth zero everywhere else.",
      "Why is unresolved. A narrower, more structured training distribution gives a position-specific Linear head more room to overfit than a translation-equivariant one -- but no run here tests that, and it is offered as a hypothesis, not a result.",
      "The gap to DQN is the finding this level actually establishes. On boards where a correct move always exists, PPO wins 13.30% and DQN wins 99.65% -- with the same board encoding, the same episode budget, comparable gradient updates, and now the same architecture.",
      "That last clause is new, and it closes a loophole. The comparison used to be confounded by DQN running a network PPO had no access to. It no longer is, and the gap is unchanged.",
      "The most likely explanation is structural rather than a missing hyperparameter. PPO is on-policy: each rollout drives a few gradient steps and is discarded. DQN keeps 20,000 transitions and re-uses every rare win for thousands of updates, which on a board where wins are scarce is most of the difference.",
      "One methodological result outlasts the rest. An earlier version of this level ran the environment chapter at about 25,000 gradient updates instead of 123,000, and reported the opposite of two findings above: no-guess boards looked mildly helpful and the discount factor looked irrelevant. Too little compute did not just shrink the effects, it reversed their sign.",
      "The architecture also bought a test nothing here could run before. This level's final weights, loaded straight into a 9x9 game with no retraining, win 1 game in 2,000 -- against 0 for a PPO trained at 9x9 from scratch. At sparse density the transferred policy is actually ahead, 1.05% against 0.35% (p = 0.012).",
      "That is a strange result stated plainly: for PPO, weights carried from 5x5 do slightly better at 9x9 than 100,000 episodes of training there. It says less about transfer than about how little the 9x9 runs ever learned. DQN's equivalent test loses roughly half its win rate without retraining (80.15% to 40.10% at standard density), which is what a normally-functioning agent looks like on the same measurement.",
      "One caveat on the whole transfer comparison: 9x9 standard is 14.8% mine density against 5x5's 20.0%, so those tests move board size and density together rather than isolating either.",
      "Every figure here is a single seed. Exactly one configuration was ever repeated -- the architecture chapter's control is a nominal replicate of the budget chapter's run, and scored 1.10% against 0.90% (p = 0.63). A 0.20-point spread from run-to-run luck alone is a useful scale to hold against every difference on this page.",
      "The structural explanation above is the best reading of this evidence, not something these runs prove.",
    ],
    intermediate: [
      "Four configurations now, all at zero wins in 2,000 -- and 11 wins across 400,000 training episodes between them.",
      "This level is closed as far as the Beginner recipe can close it. Both board settings and both networks have been run, so the remaining question is no longer which of them was untested.",
      "Compute is not the story, because it was settled at Beginner. Every run here inherits the matched configuration -- 10 PPO epochs at batch 32, ~150,000 gradient updates against DQN's 343,000 for the same 100,000 episodes.",
      "The binding constraint is depth, and it is not close. A 9x9 win needs 69 correct reveals. The best run here has a median episode of 5 and a deepest-ever of 36.",
      "It is not losing winnable games late. It never reaches a position where a win is in play at all.",
      "The board change that worked everywhere else does nothing. Making the first click safe took PPO from 1.75% to 7.90% at 5x5 and DQN from 75.00% to 80.15% at this board size; here it is 0.00% against 0.00% on identical boards.",
      "That null is unusually clean, because the change demonstrably applied -- an entire failure mode, losing on move one, was removed outright and the win rate did not move.",
      "The architecture is the one thing that changes anything measurable, and what it changes is the failure mode rather than the outcome. On the default network the policy collapses to a confident non-winning habit (entropy 0.64). With `fully_conv` it never collapses at all, holding 1.2-2.9 across both board settings.",
      "It also plays deeper and wins more often in training: 139 episodes per 100,000 reach 20+ moves against 29, and 11 training wins against 1. Every one of those gains dies before evaluation.",
      "The only non-zero evaluation cell anywhere at this board size is 0.70% at sparse density under the safe opening -- fourteen wins in 2,000, double the default network's seven, and not significant at p = 0.19.",
      "The contrast with DQN is the finding this level establishes. Same board, same episode budget, same 11-channel encoding, comparable gradient updates, and -- now -- the same `fully_conv` network: 52.75% and 80.15% against 0.00% and 0.00%.",
      "That last clause matters, because the comparison used to be confounded. DQN ran an architecture PPO had no access to, which was a legitimate objection to every number on this page. It no longer applies, and the gap is exactly as wide.",
      "Zero-shot transfer sharpens it further. DQN's 5x5 weights, loaded here untouched, score 78.25% / 40.10% / 3.30% -- roughly half what the same recipe reaches trained here, every gap significant. Retraining is worth a great deal to DQN.",
      "PPO's transferred 5x5 weights score 1.05% / 0.05% / 0.00% -- level with the runs trained here, and at sparse density nominally ahead of them (1.05% against 0.70%, p = 0.31, so not a difference this evidence carries).",
      "The asymmetry is the point regardless of that p-value. Skipping training costs DQN roughly half its win rate and costs PPO nothing measurable, because there is nothing at this board size PPO manages to learn.",
      "That is consistent with the Beginner level's structural reading -- PPO discards each rollout while DQN re-uses 20,000 stored transitions, which matters most where wins are rare -- and 9x9 is where wins are rarest. This evidence fits that explanation; it does not prove it.",
      "What would test it is a change nobody here has made: the rollout length has been 256 steps since the 5x5 runs, which cannot contain a 69-reveal trajectory. That is the first thing to vary if this level is reopened, not another board setting.",
      "All four runs are single seeds and none was tuned for 9x9. This level establishes that the Beginner recipe does not carry, not that PPO cannot be made to work at this board size.",
    ],
  },
};

/**
 * Why the architecture diagram already looks the way it does before the
 * ablation below has even started -- grounded verbatim in
 * `agents/dqn_agent.py`/`agents/ppo_agent.py`'s own module docstrings, the
 * real design history behind each fixed choice (11-channel encoding, Double
 * DQN, the replay buffer/target network, PPO's on-policy structure and
 * clipped objective). None of these are "variants" tested in the ablation --
 * they're foundations the ablation is built on top of, and the docstrings
 * are explicit that at least one of them (Huber loss + gradient clipping)
 * only exists because an even earlier, undocumented-here version of DQN
 * diverged under plain MSE loss. Shown once, above the diagram, since it
 * explains every level's pipeline below equally.
 */
export const ARCHITECTURE_RATIONALE: Record<string, string[]> = {
  DQN: [
    "11-channel one-hot encoding, not a single raw scalar channel: a hidden mask, a revealed mask, and one channel per adjacent-mine count 0-8. Minesweeper's clue numbers are categorical, not continuous, and a single scalar would put -1 (hidden) numerically adjacent to 0 (revealed, no neighbouring mines) -- the two states it matters most to tell apart. One-hot makes that distinction structural instead of something the network has to learn to disentangle. (There is no flag channel: the environment only exposes a reveal action.)",
    "Double DQN from the very first run, not vanilla DQN: plain DQN's max operator picks the best next action and evaluates it with the same network, which is disproportionately likely to overestimate. Decoupling \"which action\" (online network) from \"how good\" (target network) removes most of that bias -- though not all of it here, since the bootstrap's argmax was initially allowed to range over already-revealed cells too (see the masked-target variant below).",
    "Experience replay and a target network are also baked in from the start -- standard fixes for the correlated-samples and moving-target problems a naive DQN update has, not things this project needed to rediscover.",
    "Even with all of that, an early version of this agent diverged under plain MSE loss (loss grew from hundreds into the millions within a few thousand episodes). Switching to Huber loss plus gradient-norm clipping is what made the architecture trainable at all -- the ablation below only starts once that foundation was already stable enough to be worth comparing variants against.",
    "Every variant below is scored over 2,000 greedy episodes, not a few hundred. That is a deliberate floor rather than a default: win rates on this board sit between roughly 0.5% and 13%, so the number of wins -- not the number of episodes -- is what sets the precision. At 2,000 episodes a 2% result carries a 95% interval about 1.2 percentage points wide, narrow enough to tell a genuine 1.5x difference between two variants from noise; at a few hundred episodes the same result spans several points and any ordering between variants is unreadable.",
    "The 2,000 episodes are also the *same* 2,000 boards for every variant, drawn from one fixed evaluation seed on a fresh environment that no run trained in. Differences between the cards below are therefore differences in play, not in luck of the draw. The cost is seconds of evaluation against roughly twelve minutes of training per run, so there is no reason to economise on it.",
  ],
  PPO: [
    "Reuses the same 11-channel one-hot encoding DQN validated, for the same reason: Minesweeper's clue numbers are categorical, not continuous.",
    "On-policy by construction -- no replay buffer, no target network. Every update uses only trajectories collected under the current policy, since PPO's clipped correction is a measure of how far a new policy has drifted from the one that collected the data, and that correction stops being meaningful once the drift is large (old data can't be reused once the policy that generated it is gone).",
    "Generalized Advantage Estimation blends a low-variance one-step estimate with a high-variance multi-step estimate, giving a tunable bias/variance tradeoff that plain REINFORCE-style full-episode returns don't offer.",
    "The clipped-surrogate objective exists because policy-gradient updates without a trust-region constraint can take one bad step that collapses the policy -- there's no target network here to keep the bootstrap from moving too fast, so clipping the probability ratio between the new and old policy is PPO's substitute for that stability guarantee.",
  ],
};
