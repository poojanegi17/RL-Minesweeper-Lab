import { useState } from "react";
import { motion, type Variants } from "framer-motion";
import { Brain, ClipboardCheck, Database, Globe, Monitor, Server } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { PipelineConnector } from "@/components/research/PipelineConnector";
import { cn } from "@/lib/cn";

interface ArchNode {
  id: string;
  label: string;
  icon: typeof Monitor;
  description: string;
}

const TOP_TIER: ArchNode[] = [
  {
    id: "frontend",
    label: "React Frontend",
    icon: Monitor,
    description: "Every page, chart, and interactive explainer you're using right now -- driven entirely by real API responses, nothing mocked.",
  },
  {
    id: "backend",
    label: "FastAPI Backend",
    icon: Server,
    description: "A read-only REST layer that serves rl/results/ artifacts as clean, typed JSON. It never trains a model or writes anything.",
  },
];

const BRANCH_TIER: ArchNode[] = [
  {
    id: "environment",
    label: "RL Environment",
    icon: Globe,
    description: "The custom Gymnasium-compatible Minesweeper environment every agent trains and is evaluated against.",
  },
  {
    id: "agents",
    label: "Agents",
    icon: Brain,
    description: "Random, CSP, Q-Learning, DQN, and PPO -- five independent decision-making implementations sharing one environment interface.",
  },
  {
    id: "evaluation",
    label: "Evaluation & Replay Generation",
    icon: ClipboardCheck,
    description: "Scripts that run trained agents deterministically, score them, and record real episodes step-by-step for the replay viewer.",
  },
  {
    id: "results",
    label: "Experiment Results",
    icon: Database,
    description: "History, summaries, checkpoints, and replays -- the actual JSON/CSV files those runs wrote to disk, and the only source this whole site reads from.",
  },
];

const containerVariants: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.08 } },
};
const nodeVariants: Variants = {
  hidden: { opacity: 0, y: 12 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.35, ease: "easeOut" } },
};

/**
 * React -> FastAPI -> (Environment / Agents / Evaluation & Replay / Results).
 * Communicates this is a full system, not a notebook with a model in it.
 * Hovering or focusing a node swaps the description panel below; the
 * connectors are the same `PipelineConnector` the research pipeline already
 * uses, just re-oriented, so "data flowing downward" reads consistently
 * across the app.
 */
export function SystemArchitectureDiagram() {
  const allNodes = [...TOP_TIER, ...BRANCH_TIER];
  const [active, setActive] = useState<string>(TOP_TIER[0].id);
  const activeNode = allNodes.find((n) => n.id === active) ?? TOP_TIER[0];

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.2 }}
      className="flex flex-col items-center gap-1"
    >
      {TOP_TIER.map((node, index) => (
        <motion.div key={node.id} variants={nodeVariants} className="flex flex-col items-center">
          <ArchNodeButton node={node} isActive={active === node.id} onActivate={() => setActive(node.id)} wide />
          {index < TOP_TIER.length - 1 && <PipelineConnector orientation="vertical" active />}
        </motion.div>
      ))}

      <PipelineConnector orientation="vertical" active />

      <motion.div variants={nodeVariants} className="grid w-full grid-cols-2 gap-3 lg:grid-cols-4">
        {BRANCH_TIER.map((node) => (
          <ArchNodeButton key={node.id} node={node} isActive={active === node.id} onActivate={() => setActive(node.id)} />
        ))}
      </motion.div>

      <motion.div
        key={activeNode.id}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
        className="mt-4 max-w-2xl text-center text-sm text-text-muted"
      >
        <span className="font-medium text-heading">{activeNode.label}</span> — {activeNode.description}
      </motion.div>
    </motion.div>
  );
}

function ArchNodeButton({
  node,
  isActive,
  onActivate,
  wide,
}: {
  node: ArchNode;
  isActive: boolean;
  onActivate: () => void;
  wide?: boolean;
}) {
  return (
    <button
      type="button"
      onMouseEnter={onActivate}
      onFocus={onActivate}
      onClick={onActivate}
      className={cn("block", wide ? "w-full max-w-xs" : "w-full")}
    >
      <Card
        interactive
        className={cn(
          "flex flex-col items-center gap-2 border-white/10 bg-gradient-to-b from-surface/80 to-surface/50 py-4 text-center shadow-md shadow-black/[0.05] backdrop-blur-sm transition-colors",
          isActive && "border-primary/50 shadow-primary/10",
        )}
      >
        <span className={cn("flex h-9 w-9 items-center justify-center rounded-lg bg-current/10", isActive ? "text-primary" : "text-text-muted")}>
          <node.icon className="h-4 w-4" />
        </span>
        <span className={cn("text-xs font-medium sm:text-sm", isActive ? "text-heading" : "text-text")}>{node.label}</span>
      </Card>
    </button>
  );
}
