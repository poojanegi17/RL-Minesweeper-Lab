import { motion } from "framer-motion";
import { CONCEPT_GLOSSARY } from "@/lib/agentExplainers";

// 8 real techniques actually implemented across the 5 agents (see
// `CONCEPT_GLOSSARY`, `rl/agents/*.py`) -- one representative pick per
// algorithm family rather than every entry, so this reads as a spread of
// approaches rather than a DQN/PPO-heavy list.
const TECHNIQUES = [
  "Experience replay",
  "Target network",
  "Double DQN",
  "Generalized Advantage Estimation (GAE)",
  "Clipped-surrogate PPO",
  "Epsilon-greedy exploration",
  "Subset deduction rule",
  "Tabular Q-table",
];

export function TechniquesCloud() {
  return (
    <section className="relative z-10 mx-auto max-w-6xl px-6 py-16 md:py-20">
      <p className="text-center text-xs tracking-widest text-white/40 uppercase">
        Powered by real, implemented techniques — not a marketing list
      </p>
      <div className="mt-10 grid grid-cols-2 gap-6 sm:grid-cols-4 lg:grid-cols-8">
        {TECHNIQUES.map((technique, i) => (
          <motion.span
            key={technique}
            initial={{ opacity: 0, y: 8 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-40px" }}
            transition={{ duration: 0.4, delay: i * 0.05 }}
            title={CONCEPT_GLOSSARY[technique]}
            className="cursor-default text-center text-sm font-semibold tracking-tight text-white/50 transition-colors hover:text-white"
          >
            {technique}
          </motion.span>
        ))}
      </div>
    </section>
  );
}
