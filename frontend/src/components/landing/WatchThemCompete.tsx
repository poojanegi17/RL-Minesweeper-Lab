import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { SectionEyebrow } from "@/components/landing/LandingPrimitives";
import { AgentMindsComparison } from "@/components/home/AgentMindsComparison";

/**
 * The landing page's centerpiece, restyled around a real existing feature
 * rather than reinvented: `AgentMindsComparison` ("Different minds, same
 * game") already had four agents racing turn-by-turn on one shared board,
 * with a live roster of who's still alive -- that *is* "watch them
 * compete," so it's retrieved here rather than duplicated.
 * `AgentMindsComparison` is built from theme-aware pieces (`Card`, `Select`,
 * `Skeleton`, ...) that already resolve dark since `ThemeProvider` forces
 * `.dark` globally now -- no local `.dark` class needed here (and adding
 * one would actively break the next part: `.dark` and `.glossy-scope` both
 * set `--color-surface` at equal specificity, so putting both classes on
 * the same element makes source order decide the winner instead of intent).
 * `.glossy-scope` (see `styles/landing.css`) turns those pieces' solid
 * dark-mode surfaces -- `Card`'s own background behind the board/roster
 * included -- translucent and blurred, so the whole thing reads as glass
 * over the animated board rather than flat panels sitting on top of it.
 */
export function WatchThemCompete() {
  return (
    <section id="watch-them-compete" className="relative z-10 mx-auto max-w-6xl px-6 py-16 md:py-24">
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        className="liquid-glass relative overflow-hidden rounded-3xl border border-white/10 shadow-[0_0_120px_-20px_rgba(0,210,255,0.25)]"
      >
        <div className="flex items-center gap-2 border-b border-white/10 bg-black/40 px-4 py-3">
          <span className="h-3 w-3 rounded-full bg-[#ff5f57]" />
          <span className="h-3 w-3 rounded-full bg-[#febc2e]" />
          <span className="h-3 w-3 rounded-full bg-[#28c840]" />
          <span className="mx-auto text-xs text-white/50">RL Minesweeper Lab — Shared Board Race</span>
          <span className="rounded-full border border-white/10 px-2 py-0.5 text-[10px] tracking-wide text-white/50 uppercase">
            Live data
          </span>
        </div>

        <div className="border-b border-white/10 px-5 py-4">
          <SectionEyebrow label="Watch them compete" tag="4 agents, one shared board" />
        </div>

        <div className="glossy-scope p-5">
          <AgentMindsComparison />
        </div>

        {/* The race shows four agents side by side, which is the comparison but
         * not the reasoning. The replay viewer is where a single agent's moves
         * can be stepped through one at a time, so this points there rather
         * than trying to fit both jobs into one widget. */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/10 px-5 py-4">
          <p className="text-sm text-text-muted">
            Want to see one agent on its own? Step through a single episode move by move — every board state, and the
            reasoning behind each click.
          </p>
          <Link
            to="/replay"
            className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-white/15 px-3.5 py-2 text-sm font-medium text-text transition-colors hover:border-white/30 hover:text-white"
          >
            Watch each agent play alone
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </motion.div>
    </section>
  );
}
