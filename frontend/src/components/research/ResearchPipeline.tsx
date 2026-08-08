import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { PipelineFlowCard } from "@/components/research/PipelineFlowCard";
import { LimitationConnector } from "@/components/research/LimitationConnector";
import { ExperimentChamber } from "@/components/research/ExperimentChamber";
import { useTheme } from "@/app/ThemeProvider";
import { agentKindFromName, slugifyAgentName } from "@/lib/agentAdapters";
import { AGENT_EXPLAINERS } from "@/lib/agentExplainers";
import { AGENT_HEX } from "@/data/types";
import type { LeaderboardEntry } from "@/types/metrics";

/**
 * The five-algorithm research pipeline, in the order they were actually
 * tried. `whyAttempted` states what problem in the *previous* step motivated
 * trying this one; `limitation` is only filled in where there's real,
 * documented evidence for it (DQN's late-training instability paraphrases
 * `rl/agents/dqn_agent.py`'s own docstring; Q-Learning's paraphrases
 * `rl/evaluation/generate_replays.py`'s real exclusion of the tabular agent
 * from replay generation, since no checkpoint was persisted for it) --
 * PPO has none here because there's no equivalent documented failure to
 * point to yet, only its live leaderboard number.
 */
export const STEPS = [
  {
    agent: "Random",
    title: "Random baseline",
    whyAttempted: "Created as a baseline to understand the difficulty of Minesweeper without any intelligence at all.",
    researchQuestion: "Can we even meaningfully measure Minesweeper's difficulty without any strategy at all?",
    limitation:
      "No learning at all — no way to improve move to move, motivating a switch to explicit logical reasoning (CSP).",
    researchDecision:
      "Random gave a real difficulty floor with zero learning. The next step was adding explicit logical reasoning instead of guessing.",
  },
  {
    agent: "CSP",
    title: "CSP solver",
    whyAttempted:
      "Introduced because Minesweeper contains logical constraints that can sometimes be solved deterministically — a step up from Random's blind guessing.",
    researchQuestion: "How far can pure logical deduction get before it runs out of provably safe moves?",
    limitation:
      "Strong when a cell can be proven safe, but nothing beyond a probability guess when it can't — motivating a learned approach that improves its guesses from experience (Q-Learning).",
    researchDecision:
      "CSP proved cells safe whenever the constraints allowed it, but had nothing beyond a probability guess otherwise. The next step was a method that improves its guesses from experience instead of falling back on probability alone.",
  },
  {
    agent: "Q-Learning",
    title: "Q-Learning",
    whyAttempted:
      "Introduced to test whether a learned value function can beat a probability guess by improving from experience, before jumping straight to a neural network.",
    researchQuestion:
      "Can a learned value function, updated purely from trial and error, beat a probability guess without a neural network?",
    limitation:
      "The lookup table only knows exact board states it has already visited and generalizes to nothing new, with no checkpoint persisted to evaluate or replay here — motivating a network that generalizes across similar patterns (DQN).",
    researchDecision:
      "Q-Learning's table only knew states it had already visited and didn't generalize, with no checkpoint persisted to evaluate here. The next step was a network that generalizes across similar board patterns instead of memorizing each one.",
  },
  {
    agent: "DQN",
    title: "DQN",
    whyAttempted:
      "Introduced because rule-based deduction (CSP) and a tabular lookup (Q-Learning) both stop working once a board state falls outside what's already been proven or memorized — DQN generalizes across similar patterns instead of requiring an exact match.",
    researchQuestion:
      "Can a neural network replace a lookup table and generalize Minesweeper decisions across unseen board patterns?",
    limitation:
      "Even after tuning, loss spiked unpredictably late in training — the final snapshot could score far worse than the model achieved mid-training, motivating a look at a policy-gradient method with a different, more stable training signal (PPO).",
    researchDecision:
      "We improved over tabular learning, but training remained unstable — loss spiked unpredictably late in training. The next step was exploring PPO, a policy-gradient method with a different, more stable training signal.",
  },
  {
    agent: "PPO",
    title: "PPO",
    whyAttempted:
      "Compare a policy-gradient method against DQN's value-based approach, and see if a more stable training signal fixes DQN's late-training volatility.",
    researchQuestion:
      "Does a policy-gradient method with a clipped surrogate objective train more stably than DQN's value-based approach on the same problem?",
    limitation: "",
    researchDecision:
      "PPO is the current best-performing approach in this project — a policy-gradient method that trains directly on which action to take, rather than estimating a value for every cell.",
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
        const winRate = leaderboard.find((entry) => entry.agent === step.agent)?.win_rate ?? null;
        const strategy = AGENT_EXPLAINERS[kind].tagline;
        const isOpen = openAgent === step.agent;
        const nextStep = STEPS[index + 1];
        const nextTagline = nextStep ? AGENT_EXPLAINERS[agentKindFromName(nextStep.agent)].tagline : null;
        const leaderboardEntry = leaderboard.find((entry) => entry.agent === step.agent);

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
                    leaderboardEntry={leaderboardEntry}
                  />
                </motion.div>
              )}
            </AnimatePresence>

            {index < STEPS.length - 1 && <LimitationConnector limitation={step.limitation} active={hasEngaged} />}
          </div>
        );
      })}
    </div>
  );
}
