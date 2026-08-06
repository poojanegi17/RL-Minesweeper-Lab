import { motion, type Variants } from "framer-motion";
import { ArrowDown } from "lucide-react";
import { Card } from "@/components/ui/Card";
import type { ArchitectureConfig } from "@/lib/agentExplainers";
import { cn } from "@/lib/cn";

interface ArchitectureDiagramProps {
  architecture: ArchitectureConfig;
  /** Agent-colored icon background, e.g. `AGENT_STYLES[kind].text`. */
  accentClassName: string;
}

const containerVariants: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.12, delayChildren: 0.1 } },
};

const boxVariants: Variants = {
  hidden: { opacity: 0, y: 12 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.35, ease: "easeOut" } },
};

/**
 * A branching network diagram: one shared input box, fanning out into N
 * parallel branches each ending in its own output -- PPO's actor/critic
 * split (one observation, two networks with different jobs). Distinct from
 * `AlgorithmPipeline`, which is strictly linear; this exists specifically
 * for architectures that split.
 */
export function ArchitectureDiagram({ architecture, accentClassName }: ArchitectureDiagramProps) {
  const { input, branches } = architecture;

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="flex flex-col items-center gap-2">
      <motion.div variants={boxVariants} className="w-full max-w-md">
        <Card className="flex items-start gap-4">
          <span className={cn("flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-current/10", accentClassName)}>
            <input.icon className="h-5 w-5" />
          </span>
          <div>
            <h3 className="font-semibold text-heading">{input.label}</h3>
            <p className="mt-1 text-sm text-text-muted">{input.description}</p>
          </div>
        </Card>
      </motion.div>

      <ArrowDown className="h-4 w-4 text-text-muted" aria-hidden="true" />

      <div className="grid w-full max-w-3xl gap-6 sm:grid-cols-2">
        {branches.map((branch) => (
          <motion.div key={branch.label} variants={boxVariants} className="flex flex-col items-center gap-2">
            <Card className="flex w-full flex-col gap-3">
              <div className="flex items-start gap-3">
                <span className={cn("flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-current/10", accentClassName)}>
                  <branch.icon className="h-4 w-4" />
                </span>
                <div>
                  <h4 className="font-semibold text-heading">{branch.label}</h4>
                  <p className="mt-1 text-sm text-text-muted">{branch.description}</p>
                </div>
              </div>
            </Card>
            <ArrowDown className="h-4 w-4 text-text-muted" aria-hidden="true" />
            <span className="rounded-lg border border-dashed border-border bg-surface-hover/60 px-3 py-1.5 font-mono text-xs text-text">
              {branch.output}
            </span>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
