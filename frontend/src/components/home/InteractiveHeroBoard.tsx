import { useEffect, useState } from "react";
import { motion, type Variants } from "framer-motion";
import { ArrowRight } from "lucide-react";
import { ReplayBoard } from "@/components/replay/ReplayBoard";
import { cn } from "@/lib/cn";
import type { ReplayDetail } from "@/types/replay";

const STAGES = ["Observation", "Decision", "Action", "Reward"] as const;

type Phase = "reset" | "deciding" | "revealed";
const PHASE_DURATION_MS: Record<Phase, number> = { reset: 1100, deciding: 700, revealed: 1700 };
const REWARD_LABEL_DELAY_MS = 500;

interface InteractiveHeroBoardProps {
  /** A real recorded episode to autoplay through -- null while still loading,
   * in which case the mystery board just stays up a little longer. */
  replay: ReplayDetail | null;
}

/**
 * The hero's centerpiece: a self-running loop through one real recorded
 * episode -- a mystery board reveals cell by cell exactly as the agent
 * actually chose them, with real recorded rewards (win/mine-hit included),
 * then resets and plays again. Explains "observe -> decide -> act -> learn"
 * visually, before the visitor scrolls to anything interactive themselves.
 * Every board state and reward here is real (`ReplayStep`, see
 * `types/replay.ts`) -- nothing is scripted or invented for the animation.
 */
export function InteractiveHeroBoard({ replay }: InteractiveHeroBoardProps) {
  const [stepIndex, setStepIndex] = useState(0); // 0 = reset/mystery, 1..N = timeline[index - 1]
  const [phase, setPhase] = useState<Phase>("reset");
  const [showRewardLabel, setShowRewardLabel] = useState(false);

  const totalSteps = replay?.timeline.length ?? 0;

  useEffect(() => {
    if (!replay || totalSteps === 0) return;
    const timer = setTimeout(() => {
      if (phase === "reset") {
        setStepIndex(1);
        setPhase("deciding");
      } else if (phase === "deciding") {
        setPhase("revealed");
      } else {
        const next = stepIndex + 1;
        if (next > totalSteps) {
          setStepIndex(0);
          setPhase("reset");
        } else {
          setStepIndex(next);
          setPhase("deciding");
        }
      }
    }, PHASE_DURATION_MS[phase]);
    return () => clearTimeout(timer);
  }, [phase, stepIndex, replay, totalSteps]);

  useEffect(() => {
    if (phase !== "revealed") {
      setShowRewardLabel(false);
      return;
    }
    const timer = setTimeout(() => setShowRewardLabel(true), REWARD_LABEL_DELAY_MS);
    return () => clearTimeout(timer);
  }, [phase, stepIndex]);

  const step = replay && stepIndex > 0 ? replay.timeline[stepIndex - 1] : null;
  const previousBoard = replay ? (stepIndex > 1 ? replay.timeline[stepIndex - 2].board_state : replay.initial_board) : null;
  const revealedBoard = step ? step.board_state : null;
  // "reset" always shows the literal "?" mystery board, even though real
  // board data (all-hidden `initial_board`) technically exists -- the point
  // of this phase is the visual "we know nothing yet" beat between loops.
  const displayBoard = phase === "reset" ? null : phase === "revealed" ? revealedBoard : previousBoard;
  const isLastStep = replay ? stepIndex === totalSteps : false;
  const mineHit = phase === "revealed" && isLastStep && replay ? !replay.won : false;

  const activeStage = phase === "reset" ? "Observation" : phase === "deciding" ? "Decision" : showRewardLabel ? "Reward" : "Action";

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="relative">
        <motion.div
          className="absolute -inset-5 -z-10 rounded-[2rem] bg-primary/25 blur-2xl"
          animate={{ opacity: phase === "deciding" ? [0.3, 0.65, 0.3] : 0.18 }}
          transition={{ duration: 1.1, repeat: phase === "deciding" ? Infinity : 0, ease: "easeInOut" }}
          aria-hidden="true"
        />

        {displayBoard ? <ReplayBoard board={displayBoard} highlightedCell={step?.action ?? null} mineHit={mineHit} /> : <MysteryBoard />}

        {phase === "revealed" && step && (
          <motion.div
            key={`${stepIndex}-reward`}
            initial={{ opacity: 0, y: 6, scale: 0.75 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ type: "spring", stiffness: 420, damping: 22 }}
            className={cn(
              "absolute -top-3 -right-3 rounded-full border px-2.5 py-1 font-mono text-xs font-semibold shadow-sm",
              step.reward > 0 && "border-emerald-500/30 bg-emerald-500/15 text-emerald-600 dark:text-emerald-400",
              step.reward < 0 && "border-red-500/30 bg-red-500/15 text-red-600 dark:text-red-400",
              step.reward === 0 && "border-border bg-surface text-text-muted",
            )}
          >
            {step.reward > 0 ? `+${step.reward}` : step.reward}
          </motion.div>
        )}

        {phase === "revealed" && isLastStep && replay && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className={cn(
              "absolute inset-x-0 -bottom-3 mx-auto w-fit rounded-full border px-3 py-1 font-mono text-[11px] tracking-wide uppercase shadow-sm",
              replay.won
                ? "border-emerald-500/30 bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                : "border-border bg-surface text-text-muted",
            )}
          >
            Episode complete -- {replay.won ? "win" : "loss"}
          </motion.div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-center gap-1.5">
        {STAGES.map((stage, index) => (
          <span key={stage} className="flex items-center gap-1.5">
            <span className={cn("relative rounded-full px-2.5 py-1 text-xs font-medium", stage !== activeStage && "bg-surface-hover")}>
              {stage === activeStage && (
                <motion.span
                  layoutId="hero-stage-highlight"
                  className="absolute inset-0 rounded-full bg-primary"
                  transition={{ type: "spring", stiffness: 500, damping: 35 }}
                />
              )}
              <span className={cn("relative z-10 transition-colors duration-200", stage === activeStage ? "text-primary-foreground" : "text-text-muted")}>
                {stage}
              </span>
            </span>
            {index < STAGES.length - 1 && <ArrowRight className="h-3 w-3 shrink-0 text-text-muted" />}
          </span>
        ))}
      </div>
    </div>
  );
}

const containerVariants: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.015, delayChildren: 0.1 } },
};
const cellVariants: Variants = {
  hidden: { opacity: 0, scale: 0.6 },
  visible: { opacity: 1, scale: 1, transition: { duration: 0.25, ease: "easeOut" } },
};

/** A purely decorative all-hidden board -- shown before a real episode has
 * loaded, or briefly between loops ("we don't know anything about this
 * board yet"), with a literal "?" glyph per the hero's mockup. */
function MysteryBoard() {
  return (
    <div className="rounded-xl border border-border bg-surface p-3 shadow-sm shadow-black/[0.02]">
      <motion.div variants={containerVariants} initial="hidden" animate="visible" className="inline-grid grid-cols-5 gap-[3px]">
        {Array.from({ length: 25 }).map((_, index) => (
          <motion.div
            key={index}
            variants={cellVariants}
            className="flex h-11 w-11 items-center justify-center rounded-[5px] bg-border/70 font-mono text-base font-semibold text-text-muted shadow-[inset_1px_1px_0_rgba(255,255,255,0.25),inset_-1px_-1px_0_rgba(0,0,0,0.12)] dark:shadow-[inset_1px_1px_0_rgba(255,255,255,0.06),inset_-1px_-1px_0_rgba(0,0,0,0.3)]"
          >
            ?
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}
