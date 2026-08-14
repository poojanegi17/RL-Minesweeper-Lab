import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { GlassPillButton } from "@/components/landing/LandingPrimitives";
import { PipelineConnector } from "@/components/research/PipelineConnector";
import { STEPS } from "@/components/research/ResearchPipeline";
import { AGENT_ICONS } from "@/components/agent/agentIcons";
import { AGENT_STYLES } from "@/data/types";
import { agentKindFromName, slugifyAgentName } from "@/lib/agentAdapters";
import { headlineEyebrowClass, staticHeadlineGradient } from "@/lib/landingStyles";
import { cn } from "@/lib/cn";

/**
 * Replaces `BoardConfigurations`/`LandingFinalCTA` in the section order --
 * a single glass card, same structure `LandingFinalCTA` used (heading,
 * short line, CTA), rather than the full `ResearchJourney` milestone list
 * this held before: that's five real, tall `PipelineFlowCard`s plus a
 * heading, too much to also fit in one screen. In between, a compact
 * sneak peek -- the same five real stages (`STEPS`, reused from the
 * `/research` page rather than re-listed), just icons and a connecting
 * flow line instead of full cards, each one still a real link to its own
 * page. The full pipeline is one click away otherwise (`/research`, where
 * `ResearchJourney`'s own detailed teaser still lives if it ever gets a new
 * home) -- `min-h-screen` + centered content, same as
 * `LandingHero`/`MeetTheAgents`, so this card is the only thing on screen
 * when it's reached.
 */
export function ResearchPipelineSection() {
  return (
    <section className="relative z-10 mx-auto flex min-h-screen max-w-6xl flex-col items-center justify-center px-6 py-20">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7 }}
        className="liquid-glass relative w-full overflow-hidden rounded-3xl px-8 py-16 text-center md:py-24"
      >
        <div
          className="pointer-events-none absolute inset-0 opacity-30"
          style={{ background: "radial-gradient(600px circle at 50% 0%, rgba(255,255,255,0.15), transparent 70%)" }}
        />

        <h2 className="relative text-5xl leading-[0.95] font-semibold tracking-tight md:text-7xl">
          <span className={headlineEyebrowClass}>five algorithms · one benchmark</span>
          <span className="block text-white">Research</span>
          <span className="block" style={staticHeadlineGradient}>
            pipeline
          </span>
        </h2>
        <p className="relative mx-auto mt-6 max-w-md text-sm leading-[1.6] text-white/60">
          Five algorithms, one evolving pipeline — from a random baseline to deep reinforcement learning, each stage
          motivated by what the one before it couldn't solve.
        </p>

        <div className="relative mx-auto mt-10 flex max-w-xl items-center justify-center">
          {STEPS.map((step, index) => {
            const kind = agentKindFromName(step.agent);
            const style = AGENT_STYLES[kind];
            const Icon = AGENT_ICONS[kind];

            return (
              <div key={step.agent} className="flex items-center">
                <Link to={`/research/${slugifyAgentName(step.agent)}`} className="group flex flex-col items-center gap-2">
                  <span
                    className={cn(
                      "flex h-11 w-11 items-center justify-center rounded-full border border-white/15 bg-white/5 transition-colors group-hover:border-white/30 group-hover:bg-white/10",
                      style.text,
                    )}
                  >
                    <Icon className="h-5 w-5" />
                  </span>
                  <span className="text-[11px] font-medium text-white/60 transition-colors group-hover:text-white">
                    {step.agent}
                  </span>
                </Link>
                {index < STEPS.length - 1 && <PipelineConnector orientation="horizontal" active />}
              </div>
            );
          })}
        </div>

        <div className="relative mt-10 flex justify-center">
          <GlassPillButton label="Explore the full pipeline" to="/research" />
        </div>
      </motion.div>
    </section>
  );
}
