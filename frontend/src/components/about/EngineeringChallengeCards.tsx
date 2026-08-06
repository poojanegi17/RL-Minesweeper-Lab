import { useState } from "react";
import { AnimatePresence, motion, type Variants } from "framer-motion";
import { ChevronDown, Gamepad2, LineChart, TestTubeDiagonal } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/cn";

interface ChallengeCard {
  icon: typeof Gamepad2;
  title: string;
  teaser: string;
  points: string[];
}

/** Every bullet here names a real, already-built piece of this project (not
 * a roadmap item) -- the custom env, the ablation-family experiment
 * infrastructure, and the human-vs-AI/replay/decision-explainer UI all
 * exist elsewhere in this codebase; this card just names them plainly. */
const CHALLENGES: ChallengeCard[] = [
  {
    icon: Gamepad2,
    title: "Environment Engine",
    teaser: "A Minesweeper implementation built to be played by both people and agents.",
    points: [
      "Custom Minesweeper game engine, not a wrapped third-party library",
      "Gymnasium-compatible interface so every agent trains against the same API",
      "Configurable board size and mine count",
      "Seeded episodes for reproducible runs",
      "A designed reward system, not just win/lose",
      "Explicit episode/termination handling (win, mine hit, step limit)",
    ],
  },
  {
    icon: TestTubeDiagonal,
    title: "Experiment Infrastructure",
    teaser: "The plumbing that turns training runs into comparable, reviewable results.",
    points: [
      "Per-run experiment tracking (hyperparameters, techniques, artifacts)",
      "Per-episode metrics collection (reward, loss, win rate, and more)",
      "A separate evaluation pass, distinct from training",
      "Deterministic replay generation for real recorded episodes",
      "Ablation-family variant comparisons (baseline vs. each isolated change)",
      "Reproducible runs via recorded seeds and configuration",
    ],
  },
  {
    icon: LineChart,
    title: "Visualization Layer",
    teaser: "Making five very different decision processes legible side by side.",
    points: [
      "A human-playable board next to a live AI-playing board",
      "A full replay viewer, stepping through a real recorded episode",
      "Per-agent decision explanations (Q-values, action probabilities, CSP inference)",
      "Real training curves, not illustrative placeholders",
      "Cross-agent comparisons on the same benchmark board",
    ],
  },
];

const containerVariants: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.1 } },
};
const cardVariants: Variants = {
  hidden: { opacity: 0, y: 16 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" } },
};

/** Three expandable cards -- what actually had to be engineered beyond "an
 * agent that plays a game," each collapsed to a one-line teaser by default
 * so the section reads as a summary first, detail on demand. */
export function EngineeringChallengeCards() {
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.2 }}
      className="grid gap-4 lg:grid-cols-3"
    >
      {CHALLENGES.map((challenge) => {
        const isOpen = expanded === challenge.title;

        return (
          <motion.div key={challenge.title} variants={cardVariants}>
            <Card
              interactive
              className="flex h-full flex-col gap-3 border-white/10 bg-gradient-to-b from-surface/80 to-surface/50 shadow-lg shadow-black/[0.06] backdrop-blur-sm"
            >
              <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                <challenge.icon className="h-5 w-5" />
              </span>
              <h3 className="font-semibold text-heading">{challenge.title}</h3>
              <p className="text-sm text-text-muted">{challenge.teaser}</p>

              <button
                type="button"
                onClick={() => setExpanded(isOpen ? null : challenge.title)}
                aria-expanded={isOpen}
                className="mt-auto flex items-center gap-1 pt-2 text-sm font-medium text-primary"
              >
                {isOpen ? "Hide details" : "Show details"}
                <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", isOpen && "rotate-180")} />
              </button>

              <AnimatePresence initial={false}>
                {isOpen && (
                  <motion.ul
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.25, ease: "easeOut" }}
                    className="flex flex-col gap-1.5 overflow-hidden border-t border-border pt-3 text-sm text-text"
                  >
                    {challenge.points.map((point) => (
                      <li key={point} className="flex gap-2">
                        <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-primary" aria-hidden="true" />
                        {point}
                      </li>
                    ))}
                  </motion.ul>
                )}
              </AnimatePresence>
            </Card>
          </motion.div>
        );
      })}
    </motion.div>
  );
}
