import { motion, type Variants } from "framer-motion";
import { Flag } from "lucide-react";
import { cn } from "@/lib/cn";

type IllustrationCell =
  | { state: "hidden" }
  | { state: "flagged" }
  | { state: "revealed"; value: number };

const hidden: IllustrationCell = { state: "hidden" };
const flagged: IllustrationCell = { state: "flagged" };
const n = (value: number): IllustrationCell => ({ state: "revealed", value });

// Static, hand-authored layout for visual purposes only — not a real game state.
const BOARD: IllustrationCell[][] = [
  [hidden, hidden, n(1), n(0), n(0), n(1), hidden, hidden],
  [hidden, n(2), n(1), n(0), n(0), n(1), n(2), hidden],
  [flagged, n(2), n(0), n(0), n(0), n(0), n(1), flagged],
  [hidden, n(1), n(1), n(0), n(0), n(1), n(1), hidden],
  [hidden, hidden, n(1), flagged, n(1), hidden, hidden, hidden],
  [hidden, hidden, hidden, hidden, hidden, hidden, hidden, hidden],
];

const NUMBER_COLORS: Record<number, string> = {
  1: "text-blue-500",
  2: "text-emerald-500",
  3: "text-red-500",
  4: "text-violet-500",
  5: "text-amber-600",
  6: "text-cyan-500",
  7: "text-text",
  8: "text-text-muted",
};

const containerVariants: Variants = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.012, delayChildren: 0.1 },
  },
};

const cellVariants: Variants = {
  hidden: { opacity: 0, scale: 0.6 },
  visible: {
    opacity: 1,
    scale: 1,
    transition: { duration: 0.25, ease: "easeOut" },
  },
};

interface BoardIllustrationProps {
  className?: string;
}

/**
 * Purely decorative Minesweeper board — a fixed, hand-authored layout with
 * no interactivity, click handlers, or game state. Exists only to signal at a
 * glance what this project is about.
 */
export function BoardIllustration({ className }: BoardIllustrationProps) {
  const columns = BOARD[0]?.length ?? 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      aria-hidden="true"
      className={cn(
        "rounded-2xl border border-border bg-surface p-4 shadow-xl shadow-black/[0.06]",
        className,
      )}
    >
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="inline-grid gap-[3px]"
        style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
      >
        {BOARD.flatMap((row, r) =>
          row.map((cell, c) => (
            <motion.div
              key={`${r}-${c}`}
              variants={cellVariants}
              className={cn(
                "flex h-8 w-8 items-center justify-center rounded-[5px] font-mono text-sm font-semibold",
                cell.state === "hidden" &&
                  "bg-border/70 shadow-[inset_1px_1px_0_rgba(255,255,255,0.25),inset_-1px_-1px_0_rgba(0,0,0,0.12)] dark:shadow-[inset_1px_1px_0_rgba(255,255,255,0.06),inset_-1px_-1px_0_rgba(0,0,0,0.3)]",
                cell.state === "flagged" &&
                  "bg-border/70 shadow-[inset_1px_1px_0_rgba(255,255,255,0.25),inset_-1px_-1px_0_rgba(0,0,0,0.12)] dark:shadow-[inset_1px_1px_0_rgba(255,255,255,0.06),inset_-1px_-1px_0_rgba(0,0,0,0.3)]",
                cell.state === "revealed" &&
                  "border border-border/80 bg-background",
              )}
            >
              {cell.state === "flagged" && (
                <Flag className="h-3.5 w-3.5 fill-red-500/20 text-red-500" />
              )}
              {cell.state === "revealed" && cell.value > 0 && (
                <span className={NUMBER_COLORS[cell.value]}>
                  {cell.value}
                </span>
              )}
            </motion.div>
          )),
        )}
      </motion.div>
    </motion.div>
  );
}
