import { motion, type Variants } from "framer-motion";
import { ArrowDown, RotateCcw } from "lucide-react";
import { Card } from "@/components/ui/Card";
import type { PipelineStepConfig } from "@/lib/agentExplainers";
import { cn } from "@/lib/cn";

interface AlgorithmPipelineProps {
  steps: PipelineStepConfig[];
  /** Draws a "repeats every step" indicator after the last box -- for cyclic
   * decision processes (e.g. Q-Learning's state/action/reward/update loop). */
  loops?: boolean;
  /** Agent-colored icon background, e.g. `AGENT_STYLES[kind].text`. */
  accentClassName: string;
}

const containerVariants: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.12, delayChildren: 0.1 } },
};

const stepVariants: Variants = {
  hidden: { opacity: 0, y: 12 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.35, ease: "easeOut" } },
};

/**
 * A generic vertical box-and-arrow flow diagram: reused for CSP's deduction
 * pipeline, DQN's encode-CNN-Qvalues forward pass, Q-Learning's update
 * cycle, and Random's trivial one-step "choice." Agent-specific content
 * comes entirely from `steps` (`@/lib/agentExplainers`) -- this component
 * has no per-agent branching of its own.
 */
export function AlgorithmPipeline({ steps, loops = false, accentClassName }: AlgorithmPipelineProps) {
  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="flex flex-col items-center gap-0"
    >
      {steps.map((step, index) => (
        <motion.div key={step.title} variants={stepVariants} className="flex w-full flex-col items-center">
          <Card className="grid w-full max-w-xl grid-cols-[auto_1fr] items-start gap-4 sm:grid-cols-[auto_1fr_auto]">
            <span className={cn("flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-current/10", accentClassName)}>
              <step.icon className="h-5 w-5" />
            </span>
            <div>
              <h3 className="font-semibold text-heading">{step.title}</h3>
              <p className="mt-1 text-sm text-text-muted">{step.description}</p>
            </div>
            {step.example && (
              <div className="col-span-2 flex flex-col gap-1 rounded-lg border border-dashed border-border bg-surface-hover/60 p-3 sm:col-span-1 sm:min-w-[160px]">
                <p className="text-[11px] font-medium tracking-wide text-text-muted uppercase">Example (illustrative)</p>
                {step.example.map((row) => (
                  <div key={row.label} className="flex items-center justify-between gap-3 text-sm">
                    <span className="font-mono text-text-muted">{row.label}</span>
                    <span className={cn("font-mono font-medium", row.highlight ? "text-primary" : "text-text")}>
                      {row.value}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </Card>

          {index < steps.length - 1 && (
            <ArrowDown className="my-2 h-4 w-4 shrink-0 text-text-muted" aria-hidden="true" />
          )}
        </motion.div>
      ))}

      {loops && (
        <motion.div variants={stepVariants} className="mt-3 flex items-center gap-1.5 text-xs text-text-muted">
          <RotateCcw className="h-3.5 w-3.5" />
          Repeats every step
        </motion.div>
      )}
    </motion.div>
  );
}
