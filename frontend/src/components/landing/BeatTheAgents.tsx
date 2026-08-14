import { useState } from "react";
import { motion } from "framer-motion";
import { SectionEyebrow } from "@/components/landing/LandingPrimitives";
import { PlayableMinesweeper, type PlayableMinesweeperSummary } from "@/components/home/PlayableMinesweeper";
import { AIComparisonBoard } from "@/components/home/AIComparisonBoard";

/**
 * "Play it yourself, then watch an agent play the same game."
 *
 * `PlayableMinesweeper` and `AIComparisonBoard` were built to be paired --
 * the first reports a summary upward via `onStateChange` and the second takes
 * exactly that summary as its only prop -- but nothing had been rendering
 * them together since the landing page was rewritten. This section is the
 * pairing they were designed for, not new behaviour.
 *
 * The visitor's own game is the honest anchor for every number elsewhere on
 * this page: a 38.55% win rate means little until you have lost a few 5x5
 * boards yourself.
 */
export function BeatTheAgents() {
  // Seeded with an idle summary rather than null so `AIComparisonBoard` can
  // render its own side immediately -- the visitor should see what the agent
  // does without being required to play first.
  const [summary, setSummary] = useState<PlayableMinesweeperSummary>({
    status: "idle",
    revealedCount: 0,
    flagCount: 0,
  });

  return (
    <section id="beat-the-agents" className="relative z-10 mx-auto max-w-6xl px-6 py-16 md:py-24">
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        className="liquid-glass relative overflow-hidden rounded-3xl border border-white/10 shadow-[0_0_120px_-20px_rgba(0,210,255,0.25)]"
      >
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-5 py-4">
          <SectionEyebrow label="Try it yourself" tag="then watch an agent try the same board" />
        </div>

        <div className="glossy-scope flex flex-col gap-8 p-5">
          <PlayableMinesweeper onStateChange={setSummary} />
          <AIComparisonBoard humanSummary={summary} />
        </div>
      </motion.div>
    </section>
  );
}
