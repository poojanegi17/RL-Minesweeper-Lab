import { useState } from "react";
import { motion } from "framer-motion";
import { ArrowRight, Brain, Coins, Eye, GitBranch, Globe, MousePointerClick } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { ReplayBoard } from "@/components/replay/ReplayBoard";
import { AlgorithmPipeline } from "@/components/agents/AlgorithmPipeline";
import { cn } from "@/lib/cn";

const CONCEPTS = [
  {
    icon: Eye,
    label: "Partial observability",
    description: "No agent ever sees mine locations -- only revealed numbers and hidden cells, exactly like a human player.",
  },
  {
    icon: GitBranch,
    label: "Sequential decisions",
    description: "Every reveal changes what's known and narrows future choices -- a chain of decisions, not one single guess.",
  },
  {
    icon: Coins,
    label: "Reward feedback",
    description: "The only signal an agent learns from: a number back from the environment after each move, not a hint or a rulebook.",
  },
];

const PIPELINE_STEPS = [
  { title: "Environment", description: "The board and rules -- the one source of truth for what actually happens.", icon: Globe },
  { title: "Observation", description: "Hidden cells and revealed numbers, exactly as the agent receives them.", icon: Eye },
  { title: "Agent", description: "CSP, Q-Learning, DQN, or PPO decides what to do with that observation.", icon: Brain },
  { title: "Action", description: "One cell to reveal, chosen from everything still hidden.", icon: MousePointerClick },
  { title: "Reward", description: "The environment scores the move -- win, lose, or reveal -- and the loop repeats.", icon: Coins },
];

interface ObservationVisualizerProps {
  /** A real board state, reused from the same featured replay the hero
   * already fetches -- null while that's still loading. */
  board: number[][] | null;
}

/**
 * "How does AI see Minesweeper?" -- the same real board shown two ways side
 * by side (human tiles via the reused `ReplayBoard`, vs. the raw numeric
 * matrix an agent actually receives), with a mobile toggle where there's no
 * room for both, followed by the reused `AlgorithmPipeline` explaining the
 * decision loop that observation feeds into.
 */
export function ObservationVisualizer({ board }: ObservationVisualizerProps) {
  const [view, setView] = useState<"human" | "agent">("human");

  return (
    <div className="flex flex-col gap-10">
      <div className="flex flex-col items-center gap-4">
        <div className="inline-flex gap-1 rounded-lg border border-border bg-surface p-1 sm:hidden">
          <Button variant={view === "human" ? "primary" : "ghost"} size="sm" onClick={() => setView("human")}>
            Human View
          </Button>
          <Button variant={view === "agent" ? "primary" : "ghost"} size="sm" onClick={() => setView("agent")}>
            Agent View
          </Button>
        </div>

        {!board ? (
          <Skeleton className="h-56 w-full max-w-xl" />
        ) : (
          <div className="relative grid w-full max-w-xl gap-6 sm:grid-cols-2">
            <div className={cn("flex flex-col items-center gap-2", view !== "human" && "hidden sm:flex")}>
              <p className="text-xs font-medium tracking-[0.08em] text-text-muted uppercase">Human View</p>
              <ReplayBoard board={board} />
              <p className="text-sm text-text-muted">Humans see tiles.</p>
            </div>

            <div
              className="pointer-events-none absolute top-1/2 left-1/2 z-10 hidden -translate-x-1/2 -translate-y-1/2 sm:flex"
              aria-hidden="true"
            >
              <motion.div
                animate={{ opacity: [0.4, 1, 0.4], scale: [0.9, 1.05, 0.9] }}
                transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
                className="flex h-8 w-8 items-center justify-center rounded-full border border-primary/40 bg-surface shadow-sm"
              >
                <ArrowRight className="h-3.5 w-3.5 text-primary" />
              </motion.div>
            </div>

            <div className={cn("flex flex-col items-center gap-2", view !== "agent" && "hidden sm:flex")}>
              <p className="text-xs font-medium tracking-[0.08em] text-text-muted uppercase">Agent View</p>
              <Card className="border-primary/20 bg-gradient-to-b from-surface to-surface-hover/40 p-4 shadow-inner">
                <div className="flex flex-col gap-1.5">
                  {board.map((row, r) => (
                    <div key={r} className="flex gap-1.5">
                      {row.map((value, c) => (
                        <motion.span
                          key={c}
                          initial={{ opacity: 0, scale: 0.6 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ delay: (r * row.length + c) * 0.02, duration: 0.25, ease: "easeOut" }}
                          className={cn(
                            "flex h-6 w-6 items-center justify-center rounded font-mono text-[11px] font-semibold",
                            value === -1 ? "bg-surface-hover text-text-muted" : "bg-primary/15 text-primary",
                          )}
                        >
                          {value}
                        </motion.span>
                      ))}
                    </div>
                  ))}
                </div>
              </Card>
              <p className="text-sm text-text-muted">Agents see numbers.</p>
            </div>
          </div>
        )}
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        {CONCEPTS.map((concept) => (
          <Card key={concept.label} className="flex flex-col gap-2">
            <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <concept.icon className="h-4 w-4" />
            </span>
            <h3 className="font-semibold text-heading">{concept.label}</h3>
            <p className="text-sm text-text-muted">{concept.description}</p>
          </Card>
        ))}
      </div>

      <div>
        <p className="mb-4 text-center text-xs font-medium tracking-[0.08em] text-text-muted uppercase">
          What that observation feeds into
        </p>
        <AlgorithmPipeline steps={PIPELINE_STEPS} accentClassName="text-primary" />
      </div>
    </div>
  );
}
