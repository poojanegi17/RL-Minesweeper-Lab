import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { PipelineFlowCard } from "@/components/research/PipelineFlowCard";
import { LimitationConnector } from "@/components/research/LimitationConnector";
import { ExperimentChamber } from "@/components/research/ExperimentChamber";
import { useTheme } from "@/app/ThemeProvider";
import { agentKindFromName, slugifyAgentName } from "@/lib/agentAdapters";
import { AGENT_EXPLAINERS } from "@/lib/agentExplainers";
import { bestWinRateFor, fetchAllBoardConfigLeaderboards } from "@/lib/boardComparison";
import { useApiQuery } from "@/hooks/useApiQuery";
import { AGENT_HEX } from "@/data/types";
import type { LeaderboardEntry } from "@/types/metrics";

/**
 * The five-algorithm research pipeline, in the order they were actually
 * tried. `whyAttempted` states what problem in the *previous* step motivated
 * trying this one; `limitation` states what this step turned out not to solve.
 *
 * !! MIXED ENVIRONMENT VERSIONS -- READ BEFORE COMPARING ANY TWO STEPS !!
 * Random, CSP, and Q-Learning are quoted at **v2**: `first_click_safe="area"`,
 * where the opening click is guaranteed to open a mine-free 3x3 block and so can
 * never lose. Q-Learning now runs at 100,000 episodes under *both* board
 * distributions (its v1 grid was re-run at that budget so the two are
 * like-for-like -- the superseded 20,000-episode files are archived under
 * `rl/analysis/q_learning_budget_archive/`), so a v1/v2 gap for it is the
 * environment alone. DQN and PPO now have v2 training of their own
 * (`dqn_v2_A_baseline` at 77.25%, `ppo_v2_J_fully_conv` at 7.90%), so each agent
 * can be quoted under either distribution -- but the two still must not be mixed.
 * Those are different games, and a ~20% share of v1 episodes was decided by an
 * opening coin flip no agent could influence. Concretely: CSP's 70.35% and PPO's
 * 1.75% cannot be subtracted, nor can Q-Learning's 71.70% and DQN's 38.55%.
 *
 * The pipeline's long-standing claim that no learned agent beats CSP no longer
 * holds as stated: Q-Learning's 71.70% is statistically indistinguishable from
 * CSP's 70.35% on the same 2,000 boards (p = 0.35), and at a 200,000-episode
 * budget it reaches 73.80%, which does beat CSP (p = 0.015). Both are single-seed
 * results and want replication before the claim is rewritten in the other
 * direction. DQN goes further and beats CSP outright at 5x5 under v2 (77.25%
 * against 70.35%), though it trails CSP again at 9x9. Regenerate v2 figures with
 * `rl/evaluation/rebaseline_board_configs.py` and
 * `rl/evaluation/analyze_csp_structure.py`.
 *
 * Every figure quoted below is a measured 2,000-episode result, not a
 * paraphrase: the v1 board/density cells come from
 * `rl/evaluation/rescore_board_configs.py` and the v2 ones from
 * `rebaseline_board_configs.py`; CSP's deduction/guess counts, forced-guess
 * loss rates, and the "clicks a typical win takes" medians all come from
 * `analyze_csp_structure.py` (`rl/analysis/csp_structure_v{1,2}.json`); the
 * DQN/PPO variant figures and their significance tests from
 * `reevaluate_checkpoints.py`. Several of
 * these replaced earlier claims that turned out to be artifacts of the original
 * 200-episode protocol -- Q-Learning is *not* indistinguishable from Random, and
 * PPO's variants are indistinguishable from each other. DQN's chapter was
 * rewritten twice: first when re-scoring showed checkpointing rather than LR
 * decay to be the biggest of its five variants, and again when the instability
 * all five were chasing turned out to be an unmasked bootstrap argmax -- fixing
 * that took the same configuration from 0.55% to 11.40% and retired the
 * checkpointing conclusion outright. See the README's "Evaluation sample size"
 * and "The bootstrap mask" before editing any number here.
 *
 * PPO has no next agent to motivate, so its `researchDecision` closes the whole
 * pipeline's story instead, surfaced as `LimitationConnector`'s `conclusion`
 * rather than a hand-off. That closing claim has been rewritten: "no RL agent
 * here outperforms CSP" is retired, since Q-Learning draws level under v2 and
 * DQN passes it outright.
 *
 * Neither DQN nor PPO has an Expert run under the current recipe, so neither is
 * quoted at 16x16 anywhere in this file and `LEVEL_PIPELINE_IDS` omits that
 * level for both. Random and CSP *are* evaluated there -- they need no training
 * -- so their Expert figures below are real and stay.
 */
export const STEPS = [
  {
    agent: "Random",
    title: "Random baseline",
    whyAttempted: [
      "Every win rate needs a floor to be measured against. Random has no memory, no learning, and never reads the board — so any agent that beats it is provably doing something.",
      "It is also a check on the environment itself: if blind clicking scored well, the board would be too easy to learn anything from.",
    ],
    researchQuestion:
      "How often does a policy with no strategy at all still win, and does that floor hold up as the board grows?",
    limitation: [
      "It cannot improve. There is no state and nothing to update, so the last game is played exactly like the first.",
      "The more useful limitation is what its floor exposes: its wins are the free opening, not good clicks. A single zero-count cell can cascade across most of a small board before Random has made one real decision.",
      "That shortcut stops working as boards grow — no wins at all in 2,000 episodes at any Expert density.",
      "So the next agent has to reason about the revealed numbers instead of relying on the board being small.",
    ],
    researchDecision: [
      "The floor is real but small: 1.30% on the benchmark board, 10.20% at its easiest cell, and exactly zero at every Expert density.",
      "It also exposes the wall every later agent hits. Random actually survives slightly longer on bigger boards — 3.35 clicks at Beginner, 4.40 at Intermediate, 5.30 at Expert.",
      "But winning needs far more clicks than that: a median of 7 at Beginner, 23 at Intermediate, 78 at Expert (measured from CSP's winning games).",
      "Surviving a couple more clicks while the winning requirement grows tenfold is a losing trade. That gap is the problem the rest of the pipeline is trying to close.",
      "Next step: add explicit logical deduction, so revealed numbers become provably safe moves instead of guesses.",
    ],
  },
  {
    agent: "CSP",
    title: "CSP solver",
    whyAttempted: [
      "Random showed the board gives away a little for free. Before reaching for machine learning, the obvious question is how much of Minesweeper is simply decidable.",
      "Every revealed number is a constraint — \"exactly N of these hidden neighbours are mines\" — and constraints can be combined. That needs no learning at all.",
      "It is the honest yardstick for everything after it: if a learned agent cannot beat fixed logic, it has not earned its complexity.",
    ],
    researchQuestion:
      "How much of Minesweeper is provable rather than guessed — and does that fraction grow or shrink as the board gets bigger?",
    limitation: [
      "It does not run out of deduction so much as it runs out of information. Every loss is now a position where more than one mine layout fits the clues, so it has to bet.",
      "That still happens 2.27 times per game on the benchmark board, and 13.05% of those bets hit a mine.",
      "Worse, its fallback is only a rough estimate: it averages each constraint's count over its size rather than working out which layouts are actually possible. So even its considered gambles are not the best available ones.",
      "Nothing in fixed logic can improve that — the rules are re-derived from scratch every move and never learn from how a bet turned out.",
      "Choosing well under irreducible uncertainty is exactly what learning from outcomes might fix, which is what Q-Learning tests next.",
    ],
    researchDecision: [
      "CSP is the strongest agent in this project, and the measurement that matters is why.",
      "Bigger boards make it better, not worse. Deductions per game climb steeply — 4.9 at Beginner, 20.5 at Intermediate, 72.3 at Expert — while forced guesses barely move (2.3 → 1.6 → 1.9).",
      "So the deduction-to-guess ratio improves from 2.2:1 to 37.2:1 across the catalog. More board means more structure for the constraint graph to exploit.",
      "That is the exact opposite of what happens to a learned representation, and it is why CSP stays the bar every later agent is measured against.",
      "Mine density is what breaks it, not size: at Expert/Dense guesses climb to 5.9 per game, the ratio collapses to 9.1:1, and the win rate falls to 12.75%.",
      "Next step: a method that learns from outcomes, since CSP can never get better at the guesses it is forced into.",
    ],
  },
  {
    agent: "Q-Learning",
    title: "Q-Learning",
    whyAttempted: [
      "CSP could prove cells safe but had no way to get better at the guesses it was forced into.",
      "Q-Learning tests the opposite trade: learn what each move is worth purely from outcomes, with no rules and no neural network — so anything it achieves is down to learning alone.",
      "Keeping it tabular deliberately isolates one weakness: it cannot see two similar boards as similar. That is exactly what the next agent exists to fix.",
    ],
    researchQuestion:
      "Can a value function learned purely from trial and error beat a probability guess — and what actually limits it, the algorithm or the way states are represented?",
    limitation: [
      "The limit is coverage, not the algorithm. A board it has never seen returns all-zero values, so every move ties and it picks at random — on an unseen board this agent *is* the Random baseline.",
      "So the win rate is really a measure of how often board states repeat, and repetition runs out fast as mines are added.",
      "At the dense preset it sees 63,872 distinct states and 55.8% of them exactly once — and there, more training actively hurts: 4.15% at 50,000 episodes falls to 0.60% at 100,000.",
      "The reason is counter-intuitive but simple. A state seen once commits to a confident guess built from a single sample. A state never seen at least ties and picks at random. Extra episodes trade honest randomness for confident mistakes.",
      "Nothing transfers. Each board size and density needs its own table trained from scratch, and no table is saved, so there is no checkpoint to replay here.",
      "All of which motivates a network that recognises similar patterns instead of requiring an exact match.",
    ],
    researchDecision: [
      "This is the biggest surprise in the project: a lookup table with no generalization whatsoever reaches 71.70% [69.69–73.63] on the benchmark board.",
      "That is statistically indistinguishable from the CSP solver's 70.35% on the same 2,000 boards (p = 0.35) — and at a 200,000-episode budget it reaches 73.80%, which does beat CSP (p = 0.015). Both are single-seed and want replication.",
      "But that result belongs entirely to the safe opening, and running both environments at the same 100,000 episodes shows how completely.",
      "Same budget, different opening: 1.90% unsafe against 71.70% safe. The board distribution is worth about 70 points on its own.",
      "Same opening, different budget: with an unsafe first click, five times the training moves it from 1.10% to just 1.90%. With a safe one, the same increase is worth 30.35% to 71.70%.",
      "So this is not two causes splitting the credit. The environment is a precondition — training budget buys almost nothing until the opening is survivable, and a great deal once it is.",
      "The honest headline: Q-Learning matches CSP on first-click-safe boards, and is nowhere near it otherwise (1.90% against CSP's 43.40% on the default environment).",
      "Next step is still a network that generalizes across similar boards rather than memorizing each one.",
    ],
  },
  {
    agent: "DQN",
    title: "DQN",
    whyAttempted: [
      "CSP can prove a cell safe but never gets better at the guesses it is forced into. Q-Learning learns from outcomes but only recognises a board it has seen before.",
      "DQN targets exactly that gap: a convolutional network treats two similar boards as similar, so what it learns in one position carries to positions it has never met.",
      "It reuses the same 11-channel encoding every other agent here gets, so what is being compared is the method, not the inputs.",
    ],
    researchQuestion:
      "Can a neural network replace a lookup table and generalize Minesweeper decisions across unseen board patterns?",
    limitation: [
      "Mine density breaks it well before board size does. On protected 5x5 boards it scores 89.40% at 3 mines and 38.90% at 8 — the same weights, the same board, three extra mines.",
      "It has stopped being uniformly ahead of deduction. At 9x9 CSP leads at every density (89.95% against 80.15% at standard), where at 5x5 DQN was ahead at two of three.",
      "The largest single gain in the whole pipeline came from changing the board, not the agent — which is an uncomfortable result for a chapter about a learning algorithm.",
      "Every figure is one seed, and no configuration was replicated.",
    ],
    researchDecision: [
      "DQN is the strongest learned agent in this project: 38.55% on the original 5x5 board and 77.25% once the opening click is survivable, against Q-Learning's 1.90% on the same unprotected boards.",
      "Its first five experiments were measuring the wrong thing. Learning-rate decay, best-checkpoint deployment, reduced network capacity and the two fixes combined were all compensating for a bug — an unmasked bootstrap argmax — not for anything intrinsic to Double DQN.",
      "Fixing it changed the shape of training, not just the score: peak loss 70,741 → 5.5, average max-Q 290,825 → 10.1 against a ceiling of 10.0, and a curve that climbs instead of peaking at episode 7,500 and decaying.",
      "Then the budget mattered more than any hyperparameter: the same configuration trained four times longer went 11.40% → 21.25%. The apparent ceiling was an unfinished run.",
      "Three settings nobody had varied — reward scaling, exploration length, replay ratio — took it to 37.90%. They only work together: two of them applied without the third score 5.95%, worse than doing nothing.",
      "The architecture change was a null on win rate (38.55% against 37.90%, p = 0.67). It earns its place by making one model valid at any board size, not by winning more games.",
      "The biggest single gain came from the board rather than the agent: training on no-guess, first-click-safe layouts is worth 22.35 points on a fixed benchmark.",
      "Next step: PPO, as a genuinely different training signal — though DQN's own ceiling has still not been found.",
    ],
  },
  {
    agent: "PPO",
    title: "PPO",
    whyAttempted: [
      "DQN learns what each move is worth and acts greedily. PPO adjusts the policy directly, with a clipped objective that limits how far one update can move it.",
      "It was introduced to test whether a fundamentally different training signal — on-policy, no replay buffer, no target network — behaves differently on the same board.",
      "Same 11-channel encoding, same episode budget, and eventually the same network, so what is being compared is the algorithm rather than its inputs.",
    ],
    researchQuestion:
      "Does a policy-gradient method with a clipped surrogate objective train more stably than DQN's value-based approach on the same problem?",
    limitation: [
      "It trains stably and never becomes competitive. Nothing diverges, no run destroys its own policy — and the DQN baseline is stable too, so stability was never the differentiator.",
      "Every change made to the agent came back null. Reward shaping, checkpoint selection, four times the training budget and the network architecture all land inside the noise band, 0.80% to 1.75% on the original boards.",
      "Only the board moved it: a guaranteed-safe opening took the same network from 1.75% to 7.90%.",
      "At 9x9 it does not work at all. Four runs, both board settings, both networks — 0.00% at standard density in every one, and 11 wins across 400,000 training episodes.",
      "The reason is depth, not tuning. A 9x9 win needs 69 correct reveals; the median episode ends after 5 and the deepest ever recorded reached 36.",
      "One lever has never been pulled. The 256-step rollout is unchanged from the 5x5 runs and cannot contain a 69-reveal trajectory, which makes it the first thing to vary rather than another board setting.",
    ],
    researchDecision: [
      "PPO ends at 7.90% on a protected 5x5 board and 13.30% where a correct move always exists — its best number anywhere, against Random's 1.30%.",
      "That is roughly a tenth of the DQN trained on the same boards, with the same encoding, the same episode budget, comparable gradient updates and now the same network. The architecture confound that used to sit on this comparison has been tested away and the gap did not move.",
      "The most likely explanation is structural rather than a missing hyperparameter. PPO is on-policy: each rollout drives a few gradient steps and is discarded. DQN keeps 20,000 transitions and re-uses every rare win for thousands of updates, which on a board where wins are scarce is most of the difference.",
      "One result outlasts the rest of this level, and it is methodological. An earlier, under-resourced version of the environment chapter ran at ~25,000 gradient updates instead of ~123,000 and reported the *opposite* of two findings: no-guess boards looked mildly helpful and the discount factor looked irrelevant. Too little compute did not shrink those effects, it reversed their sign.",
      "Closing the pipeline: the project's long-standing claim that no learned agent beats CSP no longer holds. Given a survivable opening click and a 100,000-episode budget, tabular Q-Learning draws level (71.70% against 70.35%, p = 0.35) and DQN passes it outright (77.25%).",
      "What decided that was never the algorithm. Across all five agents the two changes that moved results most were spending the training budget honestly and fixing the board the agent was asked to learn from.",
      "Everything here is a single seed, and only one configuration in the project was ever replicated. Read these as one careful pass, not a settled ranking.",
    ],
  },
] as const;

interface ResearchPipelineProps {
  leaderboard: LeaderboardEntry[];
  /** Agent slug from the `/research/:agentSlug` route, if any -- opens that
   * node's chamber on first render. */
  initialAgent?: string | null;
}

export function ResearchPipeline({ leaderboard, initialAgent }: ResearchPipelineProps) {
  const { theme } = useTheme();
  const [openAgent, setOpenAgent] = useState<string | null>(
    initialAgent ? STEPS.find((step) => slugifyAgentName(step.agent) === initialAgent)?.agent ?? null : null,
  );
  const openChamberRef = useRef<HTMLDivElement>(null);
  const hasEngaged = openAgent !== null;
  // The best win rate each agent has seen across every board size/density
  // it's been evaluated at, not just the default beginner/standard board
  // (`leaderboard` prop) -- fetched once for the whole row, since a single
  // `getLeaderboard(level, density)` call already covers every agent.
  const { data: boardSnapshots } = useApiQuery(fetchAllBoardConfigLeaderboards, []);

  useEffect(() => {
    if (openAgent) {
      openChamberRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
    // Only scroll on the initial deep-link open, not on every toggle -- avoids
    // yanking the page around each time the visitor clicks a different card.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function toggle(agent: string) {
    const next = openAgent === agent ? null : agent;
    setOpenAgent(next);
    if (next) {
      requestAnimationFrame(() => openChamberRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
    }
  }

  return (
    <div className="flex flex-col">
      {STEPS.map((step, index) => {
        const kind = agentKindFromName(step.agent);
        const accentColor = AGENT_HEX[kind][theme];
        const defaultWinRate = leaderboard.find((entry) => entry.agent === step.agent)?.win_rate ?? null;
        const winRate = boardSnapshots ? (bestWinRateFor(step.agent, boardSnapshots) ?? defaultWinRate) : defaultWinRate;
        const strategy = AGENT_EXPLAINERS[kind].tagline;
        const isOpen = openAgent === step.agent;
        const nextStep = STEPS[index + 1];
        const nextTagline = nextStep ? AGENT_EXPLAINERS[agentKindFromName(nextStep.agent)].tagline : null;
        const isLastStep = index === STEPS.length - 1;

        return (
          <div key={step.agent} className="flex flex-col">
            <PipelineFlowCard
              title={step.title}
              kind={kind}
              accentColor={accentColor}
              strategy={strategy}
              winRate={winRate}
              onClick={() => toggle(step.agent)}
              isOpen={isOpen}
              index={index}
            />

            <AnimatePresence>
              {isOpen && (
                <motion.div
                  ref={openChamberRef}
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3, ease: "easeOut" }}
                  className="mt-4 overflow-hidden rounded-2xl border border-border bg-surface/60 p-5 sm:p-7"
                >
                  <ExperimentChamber
                    agentName={step.agent}
                    kind={kind}
                    accentColor={accentColor}
                    whyAttempted={step.whyAttempted}
                    researchQuestion={step.researchQuestion}
                    limitation={step.limitation}
                    researchDecision={step.researchDecision}
                    nextTagline={nextTagline}
                  />
                </motion.div>
              )}
            </AnimatePresence>

            {step.limitation && (
              <LimitationConnector
                limitation={step.limitation}
                conclusion={isLastStep ? step.researchDecision : undefined}
                active={hasEngaged}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
