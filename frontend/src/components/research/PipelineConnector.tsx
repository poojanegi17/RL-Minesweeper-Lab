import { motion } from "framer-motion";
import { cn } from "@/lib/cn";

interface PipelineConnectorProps {
  orientation: "horizontal" | "vertical";
  /** Lit up (brighter, faster dot) once the pipeline has been engaged with --
   * a subtle "progression" cue rather than every connector looking identical. */
  active?: boolean;
}

/**
 * A connector between two pipeline milestones with a small glowing dot
 * animating along it -- "data/insight flowing to the next stage." Same idea
 * as the vertical connector the old `ResearchJourney` timeline used, just
 * with a horizontal variant for the new flow-diagram layout.
 */
export function PipelineConnector({ orientation, active }: PipelineConnectorProps) {
  if (orientation === "vertical") {
    return (
      <div className="relative my-1 h-8 w-px shrink-0 bg-border" aria-hidden="true">
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

  return (
    <div className="relative mx-1 h-px min-w-6 flex-1 shrink-0 self-center bg-border" aria-hidden="true">
      <motion.span
        className={cn(
          "absolute top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full shadow-[0_0_6px_1px]",
          active ? "bg-primary shadow-primary/60" : "bg-text-muted/60 shadow-transparent",
        )}
        animate={{ left: ["0%", "90%"], opacity: [0, 1, 0] }}
        transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
      />
    </div>
  );
}
