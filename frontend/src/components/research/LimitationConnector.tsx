import { motion } from "framer-motion";
import { AlertTriangle } from "lucide-react";
import { cn } from "@/lib/cn";

interface LimitationConnectorProps {
  /** What the previous agent couldn't do -- the reason the next card exists.
   * Empty for the last step (PPO), which has no documented failure to point to yet. */
  limitation: string;
  /** Lit up once the pipeline has been engaged with, matching `PipelineConnector`. */
  active?: boolean;
}

/** A short animated line segment with a dot travelling along it, used above
 * and below the limitation box -- same "flowing to the next stage" idea as
 * `PipelineConnector`, just vertical-only since the flow card stack never
 * goes horizontal. */
function FlowLine({ active }: { active?: boolean }) {
  return (
    <div className="relative h-6 w-px shrink-0 bg-border" aria-hidden="true">
      <motion.span
        className={cn(
          "absolute left-1/2 h-1.5 w-1.5 -translate-x-1/2 rounded-full shadow-[0_0_6px_1px]",
          active ? "bg-primary shadow-primary/60" : "bg-text-muted/60 shadow-transparent",
        )}
        animate={{ top: ["0%", "90%"], opacity: [0, 1, 0] }}
        transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
      />
    </div>
  );
}

/**
 * Sits between two `PipelineFlowCard`s and states, in the flow itself, why
 * the next agent was needed -- the limitation the previous algorithm hit.
 * Text comes verbatim from `ResearchPipeline`'s `STEPS[i].limitation`, the
 * same sentence the chamber's "Research Decision" chapter already shows, just
 * surfaced here too instead of requiring a click to see it.
 */
export function LimitationConnector({ limitation, active }: LimitationConnectorProps) {
  if (!limitation) {
    return (
      <div className="flex flex-col items-center py-1">
        <FlowLine active={active} />
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-0 py-1">
      <FlowLine active={active} />
      <div className="my-2 flex w-full max-w-2xl items-start gap-2.5 rounded-lg border border-amber-500/20 bg-amber-500/[0.06] px-4 py-3">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400" aria-hidden="true" />
        <div>
          <p className="text-xs font-medium tracking-wide text-amber-600 uppercase dark:text-amber-400">Limitation — why we needed the next agent</p>
          <p className="mt-1 text-sm text-text">{limitation}</p>
        </div>
      </div>
      <FlowLine active={active} />
    </div>
  );
}
