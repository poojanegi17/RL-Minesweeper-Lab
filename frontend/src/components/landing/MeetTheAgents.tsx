import { motion } from "framer-motion";
import { AgentShowcase } from "@/components/home/AgentShowcase";
import { headlineEyebrowClass, staticHeadlineGradient } from "@/lib/landingStyles";

/**
 * Replaces `DecisionBreakdown`/`TechniquesCloud` in the section order --
 * retrieves `AgentShowcase` (the original homepage's "Meet the agents" card
 * grid, five agents that reveal how each one decides on hover) rather than
 * reinventing it, same approach as `WatchThemCompete`/`AgentMindsComparison`.
 * The heading matches `BoardConfigurations`' giant watermark treatment --
 * same gradient, same weight -- as a normal heading rather than a background
 * watermark, since this section's content is the point, not a backdrop for
 * cards floating over it. `min-h-screen` + centered content, same as
 * `LandingHero`, so this is the only thing on screen when it's reached --
 * everything after it (`BoardConfigurations`, `LandingFinalCTA`) only comes
 * into view once the visitor actually scrolls past it. `AgentShowcase`'s
 * cards carry their own `.lp-agent-card` black-glass styling directly now,
 * so no `.glossy-scope` wrapper is needed here.
 */
export function MeetTheAgents() {
  return (
    <section className="relative z-10 mx-auto flex min-h-screen max-w-6xl flex-col items-center justify-center px-6 py-20">
      <motion.h2
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7 }}
        className="text-center text-5xl leading-[0.95] font-semibold tracking-tight md:text-7xl"
      >
        <span className={headlineEyebrowClass}>05 agents · one shared benchmark</span>
        <span className="block text-white">Meet the</span>
        <span className="block" style={staticHeadlineGradient}>
          agents
        </span>
      </motion.h2>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7, delay: 0.1 }}
        className="mt-14 w-full"
      >
        <AgentShowcase />
      </motion.div>
    </section>
  );
}
