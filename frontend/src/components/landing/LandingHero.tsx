import { motion } from "framer-motion";
import { GlassPillButton } from "@/components/landing/LandingPrimitives";
import { staticHeadlineGradient } from "@/lib/landingStyles";

function scrollToId(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

export function LandingHero() {
  return (
    <section className="relative z-10 flex min-h-screen flex-col items-center justify-center px-6 py-20 text-center">
      <motion.h1
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
        className="text-4xl leading-[0.9] font-semibold tracking-tight md:text-7xl"
      >
        <span className="block text-white">Can you teach an AI</span>
        <span className="block" style={staticHeadlineGradient}>
          to play Minesweeper?
        </span>
      </motion.h1>

      <motion.p
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.5, ease: "easeOut" }}
        className="mt-8 max-w-md text-base leading-[1.5] text-white/60"
      >
        Five agents, from a random baseline to deep reinforcement learning, evaluated on the same real Minesweeper
        boards. Watch exactly how each one decides, move by move.
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.7, ease: "easeOut" }}
        className="mt-8 flex flex-col items-center gap-3"
      >
        <div className="flex flex-wrap items-center justify-center gap-3">
          <GlassPillButton label="Explore the agents" to="/agents" />
          <button
            type="button"
            onClick={() => scrollToId("watch-them-compete")}
            className="rounded-full border border-white/15 px-5 py-3 text-sm font-medium text-white transition-colors hover:bg-white/5"
          >
            Watch them compete
          </button>
        </div>
        <span className="text-xs text-white/40">5 agents · 3 board sizes · 2 mine densities · every episode recorded</span>
      </motion.div>
    </section>
  );
}
