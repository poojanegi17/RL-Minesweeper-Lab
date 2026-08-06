import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BarChart3, Dices, Network, PieChart, PlayCircle, type LucideIcon } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { EmptyState } from "@/components/ui/EmptyState";
import { ReplayBoard } from "@/components/replay/ReplayBoard";
import { AGENT_ICONS } from "@/components/agent/agentIcons";
import { AGENT_STYLES } from "@/data/types";
import { agentKindFromName } from "@/lib/agentAdapters";
import { useAgentReplay } from "@/hooks/useAgentReplay";
import { describeDecisionReason } from "@/lib/reasoning";
import { cn } from "@/lib/cn";

const AGENTS = ["CSP", "DQN", "PPO", "Random"];

/** A small per-algorithm visual motif -- CSP's constraint graph, DQN's value
 * estimate, PPO's action-probability distribution, Random's uniform draw --
 * purely decorative, so each agent's snapshot card *feels* like its own
 * approach instead of four identical layouts with a different label. */
const AGENT_MOTIFS: Record<string, { icon: LucideIcon; label: string }> = {
  CSP: { icon: Network, label: "Constraint graph" },
  DQN: { icon: BarChart3, label: "Value estimate" },
  PPO: { icon: PieChart, label: "Action distribution" },
  Random: { icon: Dices, label: "Uniform draw" },
};

/**
 * "Different minds, same game" -- pick an agent and see one real recorded
 * moment: its board, the cell it chose, and why. A single snapshot, not a
 * stepper (`AIComparisonBoard`, above, already owns full playback) -- built
 * entirely from `useAgentReplay` and `describeDecisionReason`, the same
 * reused hook/helper every other agent-reasoning view on this site uses.
 */
export function AgentMindsComparison() {
  const [agent, setAgent] = useState("CSP");
  const { replay, status, error, retry } = useAgentReplay(agent);

  const step = useMemo(() => {
    if (!replay) return null;
    return replay.timeline.find((s) => s.reasoning !== null) ?? replay.timeline[0] ?? null;
  }, [replay]);

  const reason = step ? describeDecisionReason(agent, step.reasoning) : null;
  const isLastStep = replay && step ? replay.timeline.indexOf(step) === replay.timeline.length - 1 : false;
  const mineHit = isLastStep && replay ? !replay.won : false;
  const motif = AGENT_MOTIFS[agent];

  return (
    <div className="flex flex-col gap-6">
      <p className="max-w-2xl text-sm text-text-muted">
        Each agent played its own independent episode -- not a shared board, but the same kind of decision every
        time.
      </p>

      <div className="flex flex-wrap justify-center gap-2">
        {AGENTS.map((name) => {
          const kind = agentKindFromName(name);
          const style = AGENT_STYLES[kind];
          const Icon = AGENT_ICONS[kind];
          const active = agent === name;
          return (
            <button
              key={name}
              type="button"
              onClick={() => setAgent(name)}
              aria-pressed={active}
              className={cn(
                "relative flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition-colors",
                active ? "border-primary text-heading" : "border-border bg-surface text-text-muted hover:text-text",
              )}
            >
              {active && (
                <motion.span
                  layoutId="agent-minds-active-pill"
                  className="absolute inset-0 -z-10 rounded-full bg-primary/10"
                  transition={{ type: "spring", stiffness: 500, damping: 35 }}
                />
              )}
              <span className={cn("flex h-5 w-5 items-center justify-center", style.text)}>
                <Icon className="h-4 w-4" />
              </span>
              {name}
            </button>
          );
        })}
      </div>

      {status === "loading" && <Skeleton className="h-64 w-full" />}
      {status === "error" && error && (
        <ApiErrorState error={error} onRetry={retry} title="Couldn't load this agent's replay" />
      )}
      {status === "success" && !replay && (
        <EmptyState
          icon={PlayCircle}
          title="No recorded episodes yet"
          description={`No replay has been generated for ${agent} yet.`}
        />
      )}
      <AnimatePresence mode="wait">
        {status === "success" && replay && step && (
          <motion.div
            key={agent}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.25, ease: "easeOut" }}
          >
            <Card className="relative flex flex-col items-center gap-6 overflow-hidden sm:flex-row sm:items-start sm:justify-center">
              {motif && (
                <div className="absolute top-3 right-3 flex items-center gap-1.5 rounded-full border border-border bg-surface-hover/60 px-2 py-1 text-[10px] tracking-wide text-text-muted uppercase">
                  <motif.icon className="h-3 w-3" />
                  {motif.label}
                </div>
              )}
              <ReplayBoard board={step.board_state} highlightedCell={step.action} mineHit={mineHit} />
              <div className="flex flex-col gap-2 text-center sm:text-left">
                <p className="text-xs font-medium tracking-wide text-text-muted uppercase">{agent}'s move</p>
                <p className="text-sm text-text">
                  Selected cell: <span className="font-mono">({step.action.row}, {step.action.col})</span>
                </p>
                <p className="text-sm text-text-muted">{reason ? `Reason: ${reason}` : "No reasoning recorded for this step."}</p>
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
