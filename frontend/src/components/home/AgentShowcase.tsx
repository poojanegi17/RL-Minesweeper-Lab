import { useState } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowUpRight } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { AGENT_ICONS } from "@/components/agent/agentIcons";
import { AGENT_STYLES } from "@/data/types";
import { agentKindFromName, slugifyAgentName } from "@/lib/agentAdapters";
import { cn } from "@/lib/cn";

/** Real catalog names -- `front` is the at-a-glance category, `back` is what
 * hovering/clicking reveals ("how it decides"). Short, accurate category
 * labels, not measured data, so hardcoding them here (rather than deriving
 * from `AGENT_EXPLAINERS`' longer technical taglines) keeps this card's
 * two-line reveal punchy. Routing (`slugifyAgentName`) and styling
 * (`agentKindFromName`/`AGENT_STYLES`/`AGENT_ICONS`) are still fully reused,
 * never re-derived. */
const AGENTS = [
  { name: "CSP", front: "Thinks using logic", back: "Rules and constraints" },
  { name: "Q-Learning", front: "Learns through trial and error", back: "Tabular value updates" },
  { name: "DQN", front: "Learns values from experience", back: "Q-value estimation" },
  { name: "PPO", front: "Learns a decision policy", back: "Action probabilities" },
  { name: "Random", front: "Picks at random", back: "No learning baseline" },
];

const containerVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.08 } },
};
const cardVariants = {
  hidden: { opacity: 0, y: 14 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.35, ease: "easeOut" as const } },
};

/** "Meet the agents" -- five cards that reveal how each one decides on
 * hover or tap, each linking to its real, live-data agent page. */
export function AgentShowcase() {
  const [revealed, setRevealed] = useState<string | null>(null);

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.2 }}
      className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5"
    >
      {AGENTS.map((agent) => {
        const kind = agentKindFromName(agent.name);
        const style = AGENT_STYLES[kind];
        const Icon = AGENT_ICONS[kind];
        const isRevealed = revealed === agent.name;

        return (
          <motion.div key={agent.name} variants={cardVariants}>
          <Card
            interactive
            onMouseEnter={() => setRevealed(agent.name)}
            onMouseLeave={() => setRevealed((r) => (r === agent.name ? null : r))}
            onClick={() => setRevealed((r) => (r === agent.name ? null : agent.name))}
            className="flex cursor-pointer flex-col gap-3"
          >
            <span className={cn("flex h-9 w-9 items-center justify-center rounded-lg bg-current/10", style.text)}>
              <Icon className="h-4 w-4" />
            </span>
            <div>
              <h3 className="font-semibold text-heading">{agent.name}</h3>
              <p className="mt-1 text-sm text-text-muted">{agent.front}</p>
            </div>

            <AnimatePresence>
              {isRevealed && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden border-t border-border pt-3"
                >
                  <p className="text-xs text-text-muted">How it decides</p>
                  <p className="mt-0.5 text-sm font-medium text-heading">{agent.back}</p>
                </motion.div>
              )}
            </AnimatePresence>

            <Link
              to={`/agents/${slugifyAgentName(agent.name)}`}
              onClick={(e) => e.stopPropagation()}
              className="mt-auto flex items-center gap-1 pt-2 text-sm font-medium text-primary hover:underline"
            >
              View agent
              <ArrowUpRight className="h-3.5 w-3.5" />
            </Link>
          </Card>
          </motion.div>
        );
      })}
    </motion.div>
  );
}
