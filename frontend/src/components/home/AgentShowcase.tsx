import { useState } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowUpRight } from "lucide-react";
import { AGENT_STYLES } from "@/data/types";
import { agentKindFromName, slugifyAgentName } from "@/lib/agentAdapters";
import { cn } from "@/lib/cn";

/** Real catalog names -- `front` is the at-a-glance category, `back` is what
 * hovering/clicking reveals ("how it decides"). Short, accurate category
 * labels, not measured data, so hardcoding them here (rather than deriving
 * from `AGENT_EXPLAINERS`' longer technical taglines) keeps this card's
 * two-line reveal punchy. Routing (`slugifyAgentName`) and styling
 * (`agentKindFromName`/`AGENT_STYLES`) are still fully reused, never
 * re-derived. */
const AGENTS = [
  { name: "CSP", front: "Thinks using logic", back: "Rules and constraints" },
  { name: "Q-Learning", front: "Learns through trial and error", back: "Tabular value updates" },
  { name: "DQN", front: "Learns values from experience", back: "Q-value estimation" },
  { name: "PPO", front: "Learns a decision policy", back: "Action probabilities" },
  { name: "Random", front: "Picks at random", back: "No learning baseline" },
];

const cardVariants = {
  hidden: { opacity: 0, y: 14 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.35, ease: "easeOut" as const } },
};

/** "Meet the agents" -- all five cards side by side, no sliding. `.lp-agent-card`
 * (see `styles/landing.css`) gives them the same black-glass identity as
 * `BoardConfigurations`' pricing cards. Per-agent identity colors
 * (`AGENT_STYLES`) live on the card's own name text rather than an icon
 * badge -- still real information (which agent), just carried differently
 * now that the icon's gone. */
export function AgentShowcase() {
  const [revealed, setRevealed] = useState<string | null>(null);

  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
      {AGENTS.map((agent) => {
        const kind = agentKindFromName(agent.name);
        const style = AGENT_STYLES[kind];
        const isRevealed = revealed === agent.name;

        return (
          <motion.div
            key={agent.name}
            variants={cardVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.3 }}
            whileHover={{ scale: 1.04, y: -10 }}
            transition={{ type: "spring", stiffness: 300, damping: 22 }}
          >
            <div
              onMouseEnter={() => setRevealed(agent.name)}
              onMouseLeave={() => setRevealed((r) => (r === agent.name ? null : r))}
              onClick={() => setRevealed((r) => (r === agent.name ? null : agent.name))}
              className="lp-agent-card flex h-full min-h-[480px] cursor-pointer flex-col gap-3"
            >
              <h3 className={cn("text-2xl font-bold tracking-tight", style.text)}>{agent.name}</h3>
              <p className="text-sm text-white/60">{agent.front}</p>

              <AnimatePresence>
                {isRevealed && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden border-t border-white/10 pt-3"
                  >
                    <p className="text-xs text-white/60">How it decides</p>
                    <p className="mt-0.5 text-sm font-medium text-white">{agent.back}</p>
                  </motion.div>
                )}
              </AnimatePresence>

              <Link
                to={`/agents/${slugifyAgentName(agent.name)}`}
                onClick={(e) => e.stopPropagation()}
                className="mt-auto flex items-center gap-1 pt-2 text-sm font-medium text-[#00d2ff] hover:underline"
              >
                View agent
                <ArrowUpRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
