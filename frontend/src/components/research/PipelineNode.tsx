import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { AGENT_ICONS } from "@/components/agent/agentIcons";
import { AGENT_STYLES } from "@/data/types";
import { formatPercent } from "@/lib/experimentAdapters";
import type { AgentKind } from "@/data/types";
import { cn } from "@/lib/cn";

interface PipelineNodeProps {
  title: string;
  kind: AgentKind;
  accentColor: string;
  winRate: number | null;
  /** Overrides the win-rate line with a short caption instead -- used by
   * `about/DesignPhilosophyProgression`, where the row is a one-line
   * rationale teaser rather than a live leaderboard figure. */
  subtitle?: string;
  /** Link mode -- used by the Home teaser, navigates instead of expanding. */
  to?: string;
  /** Toggle mode -- used by the full pipeline, expands `ExperimentChamber` below. */
  onClick?: () => void;
  isOpen?: boolean;
  index?: number;
}

/**
 * One milestone card in the algorithm pipeline -- shared between the Home
 * teaser (`to`, a plain `Link`) and the full `/research` explorer (`onClick`
 * + `isOpen`, a toggle button), so the visual "milestone" never has two
 * separate implementations.
 */
export function PipelineNode({ title, kind, accentColor, winRate, subtitle, to, onClick, isOpen, index = 0 }: PipelineNodeProps) {
  const style = AGENT_STYLES[kind];
  const Icon = AGENT_ICONS[kind];

  const content = (
    <Card
      interactive
      className={cn(
        "relative flex w-full flex-col items-center gap-2 overflow-hidden text-center transition-colors sm:w-40",
        isOpen && "border-primary/50 shadow-md shadow-primary/10",
      )}
      style={isOpen ? { borderColor: `${accentColor}80` } : undefined}
    >
      <span className={cn("flex h-11 w-11 items-center justify-center rounded-lg bg-current/10", style.text)}>
        <Icon className="h-5 w-5" />
      </span>
      <h3 className="text-sm font-semibold text-heading">{title}</h3>
      {subtitle ? (
        <p className="text-xs text-text-muted">{subtitle}</p>
      ) : winRate != null ? (
        <span className="font-mono text-sm font-semibold tabular-nums" style={{ color: accentColor }}>
          {formatPercent(winRate)}
          <span className="ml-1 text-[10px] font-normal text-text-muted">win rate</span>
        </span>
      ) : (
        <span className="text-[11px] text-text-muted">no data yet</span>
      )}
      {onClick && (
        <ChevronDown
          className={cn("h-3.5 w-3.5 text-text-muted transition-transform", isOpen && "rotate-180 text-primary")}
          aria-hidden="true"
        />
      )}
    </Card>
  );

  const motionProps = {
    initial: { opacity: 0, y: 16 },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true, amount: 0.4 },
    whileHover: { y: -3 },
    transition: { duration: 0.4, ease: "easeOut" as const, delay: index * 0.06 },
  };

  if (to) {
    return (
      <motion.div {...motionProps}>
        <Link to={to} className="block">
          {content}
        </Link>
      </motion.div>
    );
  }

  return (
    <motion.div {...motionProps}>
      <button
        type="button"
        onClick={onClick}
        aria-expanded={isOpen}
        className="block w-full rounded-xl text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60"
      >
        {content}
      </button>
    </motion.div>
  );
}
