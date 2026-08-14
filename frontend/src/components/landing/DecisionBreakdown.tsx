import { motion } from "framer-motion";
import { SectionEyebrow } from "@/components/landing/LandingPrimitives";
import { Skeleton } from "@/components/ui/Skeleton";
import { getReplays } from "@/api/replays";
import { useApiQuery } from "@/hooks/useApiQuery";

// Same live config the replay viewer centerpiece uses -- see its comment for
// why this (level, density) pair specifically (every agent has a genuine,
// directly-comparable `board_result` entry there).
const LEVEL = "beginner";
const DENSITY = "sparse";

const CHIPS = ["Per-step reasoning", "Same board every agent", "Every episode recorded", "2,000-episode evaluation"];

interface Bucket {
  label: string;
  count: number;
  tone: string;
}

export function DecisionBreakdown() {
  const { data, status } = useApiQuery(() => getReplays(LEVEL, DENSITY), []);

  const buckets: Bucket[] | null = (() => {
    if (!data || data.length === 0) return null;
    const wins = data.filter((r) => r.won);
    const losses = data.filter((r) => !r.won);
    const avgWinLength = wins.length > 0 ? wins.reduce((sum, r) => sum + r.steps, 0) / wins.length : 0;
    const longLosses = losses.filter((r) => r.steps > avgWinLength);
    const fastLosses = losses.filter((r) => r.steps <= avgWinLength);
    return [
      { label: "Wins", count: wins.length, tone: "#ffffff" },
      { label: `Losses past ${avgWinLength.toFixed(0)} moves`, count: longLosses.length, tone: "#e5e5e5" },
      { label: `Losses within ${avgWinLength.toFixed(0)} moves`, count: fastLosses.length, tone: "#a3a3a3" },
    ];
  })();

  return (
    <section className="relative z-10 mx-auto grid max-w-6xl gap-10 px-6 py-20 md:grid-cols-2 md:gap-16 md:py-28">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7 }}
      >
        <SectionEyebrow label="Decisions" tag="Recorded, not simulated" />
        <h2 className="mt-5 text-3xl leading-[1.02] font-semibold tracking-tight md:text-5xl">
          See exactly how
          <br />
          it decided.
        </h2>
        <p className="mt-6 max-w-md text-base leading-[1.6] text-white/60">
          Every recorded episode logs the board state and the agent's reasoning at each step — a Q-value, an action
          probability, or a logical deduction, depending on the agent. Nothing here is replayed after the fact; it's
          exactly what was recorded during evaluation.
        </p>
        <div className="mt-6 flex flex-wrap gap-2">
          {CHIPS.map((chip) => (
            <span key={chip} className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs text-white/70">
              {chip}
            </span>
          ))}
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7, delay: 0.1 }}
        className="liquid-glass rounded-2xl p-5"
      >
        <p className="text-xs text-white/50">
          Beginner · 5×5 · 3 mines · {data?.length ?? "—"} recorded episodes across 5 agents
        </p>

        {status === "loading" && (
          <div className="mt-4 flex flex-col gap-3">
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        )}

        {buckets && (
          <div className="mt-4 flex flex-col gap-3">
            {buckets.map((bucket) => (
              <div key={bucket.label} className="liquid-glass flex items-center justify-between rounded-lg p-3">
                <span className="text-sm text-white/80">{bucket.label}</span>
                <span className="font-mono text-lg font-semibold" style={{ color: bucket.tone }}>
                  {bucket.count}
                </span>
              </div>
            ))}
          </div>
        )}
      </motion.div>
    </section>
  );
}
