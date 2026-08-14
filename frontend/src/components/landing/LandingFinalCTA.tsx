import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ChevronRight } from "lucide-react";
import { GlassPillButton } from "@/components/landing/LandingPrimitives";

export function LandingFinalCTA() {
  return (
    <section className="relative z-10 mx-auto max-w-6xl px-6 py-20 md:py-32">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7 }}
        className="liquid-glass relative overflow-hidden rounded-3xl px-8 py-16 text-center md:py-24"
      >
        <div
          className="pointer-events-none absolute inset-0 opacity-30"
          style={{ background: "radial-gradient(600px circle at 50% 0%, rgba(255,255,255,0.15), transparent 70%)" }}
        />

        <h2 className="relative text-4xl leading-[1.02] font-semibold tracking-tight md:text-6xl">
          Watch it think.
          <br />
          Then judge for yourself.
        </h2>
        <p className="relative mx-auto mt-6 max-w-md text-sm leading-[1.6] text-white/60">
          Five agents, real evaluation runs, every recorded episode available to step through — no cherry-picked
          highlight reels.
        </p>
        <div className="relative mt-8 flex flex-wrap items-center justify-center gap-3">
          <GlassPillButton label="Explore the agents" to="/agents" />
          <Link to="/research" className="inline-flex items-center gap-1.5 rounded-full border border-white/15 px-5 py-3 text-sm font-medium text-white transition-colors hover:bg-white/5">
            View the research
            <ChevronRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </motion.div>
    </section>
  );
}
