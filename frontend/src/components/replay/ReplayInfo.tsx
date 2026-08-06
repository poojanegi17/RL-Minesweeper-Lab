import { Badge } from "@/components/ui/Badge";
import type { ReplayDetail, ReplayStep } from "@/types/replay";
import { cn } from "@/lib/cn";

interface ReplayInfoProps {
  replay: ReplayDetail;
  /** The step currently shown, or null at "step 0" (before any action). */
  currentStep: ReplayStep | null;
}

/** Episode summary (result/steps/reward) plus the current step's agent-specific
 * reasoning, where one was recorded -- see `ReasoningDisplay` for the per-agent shapes. */
export function ReplayInfo({ replay, currentStep }: ReplayInfoProps) {
  return (
    <div className="flex flex-col gap-6">
      <dl className="grid grid-cols-3 gap-4">
        <div>
          <dd>
            <Badge variant={replay.won ? "neutral" : "outline"} className={cn(replay.won && "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400")}>
              {replay.won ? "WIN" : "LOSS"}
            </Badge>
          </dd>
          <dt className="mt-1.5 text-xs text-text-muted">Result</dt>
        </div>
        <div>
          <dd className="font-mono text-xl font-semibold text-heading">{replay.steps}</dd>
          <dt className="mt-1 text-xs text-text-muted">Steps</dt>
        </div>
        <div>
          <dd className="font-mono text-xl font-semibold text-heading">{replay.total_reward.toLocaleString()}</dd>
          <dt className="mt-1 text-xs text-text-muted">Reward</dt>
        </div>
      </dl>

      <div className="border-t border-border pt-4">
        <h3 className="mb-2 text-xs font-medium tracking-wide text-text-muted uppercase">
          Current step
        </h3>
        {currentStep ? (
          <div className="flex flex-col gap-3">
            <p className="text-sm text-text">
              Selected action: <span className="font-mono">({currentStep.action.row}, {currentStep.action.col})</span>
            </p>
            <ReasoningDisplay agent={replay.agent} reasoning={currentStep.reasoning} />
          </div>
        ) : (
          <p className="text-sm text-text-muted">Initial board -- no action taken yet.</p>
        )}
      </div>
    </div>
  );
}

function ReasoningDisplay({ agent, reasoning }: { agent: string; reasoning: Record<string, unknown> | null }) {
  if (!reasoning) {
    return <p className="text-sm text-text-muted">No explanation metadata recorded</p>;
  }

  if (agent === "DQN" && typeof reasoning.q_value === "number") {
    return <ReasoningRow label="Q-value" value={reasoning.q_value.toFixed(3)} />;
  }

  if (agent === "PPO" && typeof reasoning.action_probability === "number") {
    return <ReasoningRow label="Action probability" value={`${(reasoning.action_probability * 100).toFixed(1)}%`} />;
  }

  if (agent === "CSP") {
    const inference = typeof reasoning.inference === "string" ? reasoning.inference : null;
    const remainingMines = typeof reasoning.remaining_mines === "number" ? reasoning.remaining_mines : null;
    const constraintCells = Array.isArray(reasoning.constraint_cells) ? (reasoning.constraint_cells as number[][]) : null;
    const mineProbability = typeof reasoning.mine_probability === "number" ? reasoning.mine_probability : null;

    return (
      <div className="flex flex-col gap-1.5">
        {constraintCells && constraintCells.length > 0 && (
          <ReasoningRow
            label="Constraint detected"
            value={constraintCells.map(([r, c]) => `(${r}, ${c})`).join(", ")}
          />
        )}
        {remainingMines !== null && <ReasoningRow label="Remaining mines" value={String(remainingMines)} />}
        {mineProbability !== null && <ReasoningRow label="Mine probability" value={`${(mineProbability * 100).toFixed(1)}%`} />}
        {inference && <p className="text-sm text-text-muted">Inference: {inference}</p>}
      </div>
    );
  }

  // Recognized as *some* recorded reasoning, but not one of the known
  // agent shapes above -- show it plainly rather than silently dropping
  // real data or pretending it matches a shape it doesn't.
  return (
    <pre className="overflow-x-auto rounded-lg bg-surface-hover p-2 font-mono text-xs text-text-muted">
      {JSON.stringify(reasoning, null, 2)}
    </pre>
  );
}

function ReasoningRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-text-muted">{label}</span>
      <span className="font-mono font-medium text-text">{value}</span>
    </div>
  );
}
