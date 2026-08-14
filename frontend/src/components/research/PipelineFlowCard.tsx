import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { AGENT_ICONS } from "@/components/agent/agentIcons";
import { AGENT_STYLES } from "@/data/types";
import { formatPercent } from "@/lib/experimentAdapters";
import type { AgentKind } from "@/data/types";
import { cn } from "@/lib/cn";

interface PipelineFlowCardProps {
  title: string;
  kind: AgentKind;
  accentColor: string;
  /** The algorithm's one-line "how it decides" tagline (`AGENT_EXPLAINERS[kind].tagline`) --
   * shown directly under the name so the strategy is visible without opening the chamber. */
  strategy: string;
  /** The best win rate seen for this agent across every board size/density
   * it's been evaluated at (see `lib/boardComparison.ts`), not just the
   * default beginner/standard board -- the same "best" the chamber's own
   * `BoardConfigComparisonTable` concludes with. */
  winRate: number | null;
  /** Link mode -- used by the Home teaser (`ResearchJourney`), navigates to
   * `/research/{slug}` instead of expanding a chamber in place. Mutually
   * exclusive with `onClick`/`isOpen`, mirroring `PipelineNode`'s existing
   * `to` vs. `onClick` split. */
  to?: string;
  onClick?: () => void;
  isOpen?: boolean;
  index: number;
}

/**
 * One full-width milestone card in the vertical research flow -- name and
 * strategy always visible. On `/research` (`onClick`/`isOpen`), click expands
 * its `ExperimentChamber` directly beneath it; on the Home teaser (`to`), it's
 * a plain link into that same chamber's page. Distinct from the compact
 * `PipelineNode` (still used by `DesignPhilosophyProgression`'s horizontal
 * row), since that layout wants a smaller footprint than this one.
 */
export function PipelineFlowCard({ title, kind, accentColor, strategy, winRate, to, onClick, isOpen, index }: PipelineFlowCardProps) {
  const style = AGENT_STYLES[kind];
  const Icon = AGENT_ICONS[kind];

  const content = (
    <Card
      interactive
      className={cn("flex w-full items-center gap-4 transition-colors sm:gap-5", isOpen && "border-primary/50 shadow-md shadow-primary/10")}
      style={isOpen ? { borderColor: `${accentColor}80` } : undefined}
    >
      <span className={cn("flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-current/10", style.text)}>
        <Icon className="h-6 w-6" />
      </span>

      <div className="min-w-0 flex-1">
        <h3 className="text-lg font-semibold text-heading">{title}</h3>
        <p className="mt-0.5 text-sm text-text-muted">{strategy}</p>
      </div>

      <div className="flex shrink-0 items-center gap-3 sm:gap-4">
        {winRate != null ? (
          <span className="hidden font-mono text-sm font-semibold tabular-nums sm:inline" style={{ color: accentColor }}>
            {formatPercent(winRate)}
            <span className="ml-1 text-[10px] font-normal text-text-muted">best win rate</span>
          </span>
        ) : (
          <span className="hidden text-[11px] text-text-muted sm:inline">no data yet</span>
        )}
        {onClick && (
          <ChevronDown className={cn("h-4 w-4 text-text-muted transition-transform", isOpen && "rotate-180 text-primary")} aria-hidden="true" />
        )}
      </div>
    </Card>
  );

  const motionProps = {
    initial: { opacity: 0, y: 16 },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true, amount: 0.3 },
    transition: { duration: 0.4, ease: "easeOut" as const, delay: index * 0.05 },
  };

  if (to) {
    return (
      <motion.div {...motionProps}>
        <Link to={to} className="block w-full rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60">
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
        className="block w-full rounded-2xl text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60"
      >
        {content}
      </button>
    </motion.div>
  );
}
