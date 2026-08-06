import { useEffect, useState } from "react";
import { ArrowRight, ChevronLeft, ChevronRight, Eye, EyeOff, PlayCircle } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { Skeleton } from "@/components/ui/Skeleton";
import { ReplayBoard } from "@/components/replay/ReplayBoard";
import { useAgentReplay } from "@/hooks/useAgentReplay";
import type { DecisionExampleConfig } from "@/lib/agentExplainers";
import type { ReplayStep } from "@/types/replay";

interface DecisionExampleProps {
  config: DecisionExampleConfig;
}

/**
 * The page's interactive centerpiece. Two modes (`@/lib/agentExplainers`):
 *
 * - `"replay"`: fetches a real recorded episode for this agent (`GET
 *   /api/replays`, then `GET /api/replays/{id}`) and lets the user step
 *   through it, revealing the exact reasoning metadata that step actually
 *   recorded -- never a fabricated or hypothetical board state.
 * - `"illustrative"`: for agents with no recorded reasoning to show
 *   (Q-Learning writes no replay/reasoning artifacts at all), a clearly
 *   captioned worked example of the real update formula -- see
 *   `AlgorithmPipeline`'s identical "(illustrative)" convention.
 */
export function DecisionExample({ config }: DecisionExampleProps) {
  if (config.mode === "illustrative") {
    return <IllustrativeExample config={config} />;
  }
  return <ReplayDecisionExample agentName={config.agentName} />;
}

function IllustrativeExample({ config }: { config: Extract<DecisionExampleConfig, { mode: "illustrative" }> }) {
  return (
    <Card className="flex flex-col gap-5">
      <div>
        <p className="text-xs font-medium tracking-wide text-text-muted uppercase">Worked example</p>
        <p className="mt-1.5 font-mono text-sm text-heading">{config.formula}</p>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-3 py-2">
        {config.rows.map((row, index) => (
          <div key={row.label} className="flex items-center gap-3">
            <div className="rounded-xl border border-border bg-surface-hover/60 px-4 py-3 text-center">
              <p className="text-xs text-text-muted">{row.label}</p>
              <p className="mt-1 font-mono text-lg font-semibold text-heading">{row.value}</p>
            </div>
            {index < config.rows.length - 1 && <ArrowRight className="h-4 w-4 shrink-0 text-text-muted" />}
          </div>
        ))}
      </div>

      <p className="text-xs text-text-muted">{config.caption}</p>
    </Card>
  );
}

function ReplayDecisionExample({ agentName }: { agentName: string }) {
  const { replay, status, error, retry } = useAgentReplay(agentName);

  const [stepIndex, setStepIndex] = useState(0);
  const [showReasoning, setShowReasoning] = useState(false);

  useEffect(() => {
    setStepIndex(0);
    setShowReasoning(false);
  }, [replay?.id]);

  if (status === "loading") {
    return <Skeleton className="h-72 w-full" />;
  }
  if (status === "error" && error) {
    return <ApiErrorState error={error} onRetry={retry} title="Couldn't load an example" />;
  }
  if (!replay) {
    return (
      <EmptyState
        icon={PlayCircle}
        title="No recorded episodes yet"
        description={`No replay has been generated for ${agentName} yet -- run generate_replays.py to create one.`}
      />
    );
  }

  const step: ReplayStep | null = stepIndex === 0 ? null : replay.timeline[stepIndex - 1];
  const board = step ? step.board_state : replay.initial_board;
  const isLastStep = stepIndex === replay.steps;
  const mineHit = isLastStep && !replay.won;

  return (
    <Card className="flex flex-col gap-6 lg:flex-row lg:items-start">
      <div className="flex flex-col items-center gap-3">
        <ReplayBoard board={board} highlightedCell={step?.action ?? null} mineHit={mineHit} />
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setStepIndex((i) => Math.max(0, i - 1))}
            disabled={stepIndex === 0}
          >
            <ChevronLeft className="h-3.5 w-3.5" />
            Prev
          </Button>
          <span className="font-mono text-xs text-text-muted">
            Step {stepIndex} / {replay.steps}
          </span>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => setStepIndex((i) => Math.min(replay.steps, i + 1))}
            disabled={stepIndex === replay.steps}
          >
            Next
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-3">
        <p className="text-sm text-text-muted">
          A real recorded episode ({replay.won ? "win" : "loss"}, {replay.steps} steps) -- exactly what {agentName} saw
          and chose, move by move.
        </p>
        {step ? (
          <>
            <p className="text-sm text-text">
              Selected action: <span className="font-mono">({step.action.row}, {step.action.col})</span>
            </p>
            <Button variant="secondary" size="sm" className="w-fit" onClick={() => setShowReasoning((v) => !v)}>
              {showReasoning ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
              {showReasoning ? "Hide reasoning" : "Show reasoning"}
            </Button>
            {showReasoning && <ReasoningPanel agent={agentName} reasoning={step.reasoning} />}
          </>
        ) : (
          <p className="text-sm text-text-muted">Initial board -- step forward to see its first move.</p>
        )}
      </div>
    </Card>
  );
}

/** Formats the same per-agent `reasoning` shapes `ReplayInfo` renders in the
 * full Replay Viewer -- kept as its own small copy here rather than a shared
 * import, since the two components live in different feature folders and
 * this one's layout (a single revealed step, not a scrubbable timeline) is
 * intentionally simpler. */
function ReasoningPanel({ agent, reasoning }: { agent: string; reasoning: Record<string, unknown> | null }) {
  if (!reasoning) {
    return <p className="text-sm text-text-muted">No explanation metadata recorded for this step.</p>;
  }

  if (agent === "DQN" && typeof reasoning.q_value === "number") {
    return (
      <div className="flex flex-col gap-1.5 rounded-lg border border-border bg-surface-hover/60 p-3">
        <ReasoningRow label="Q-value" value={reasoning.q_value.toFixed(3)} />
      </div>
    );
  }

  if (agent === "PPO" && typeof reasoning.action_probability === "number") {
    return (
      <div className="flex flex-col gap-1.5 rounded-lg border border-border bg-surface-hover/60 p-3">
        <ReasoningRow label="Action probability" value={`${(reasoning.action_probability * 100).toFixed(1)}%`} />
      </div>
    );
  }

  if (agent === "CSP") {
    const inference = typeof reasoning.inference === "string" ? reasoning.inference : null;
    const remainingMines = typeof reasoning.remaining_mines === "number" ? reasoning.remaining_mines : null;
    const constraintCells = Array.isArray(reasoning.constraint_cells) ? (reasoning.constraint_cells as number[][]) : null;
    const mineProbability = typeof reasoning.mine_probability === "number" ? reasoning.mine_probability : null;

    return (
      <div className="flex flex-col gap-1.5 rounded-lg border border-border bg-surface-hover/60 p-3">
        {constraintCells && constraintCells.length > 0 && (
          <ReasoningRow label="Constraint" value={constraintCells.map(([r, c]) => `(${r}, ${c})`).join(", ")} />
        )}
        {remainingMines !== null && <ReasoningRow label="Remaining mines" value={String(remainingMines)} />}
        {mineProbability !== null && <ReasoningRow label="Mine probability" value={`${(mineProbability * 100).toFixed(1)}%`} />}
        {inference && <ReasoningRow label="Inference" value={inference} />}
      </div>
    );
  }

  // Recognized as *some* recorded reasoning, but not one of the known agent
  // shapes above -- show it plainly rather than silently dropping real data.
  return (
    <pre className="overflow-x-auto rounded-lg bg-surface-hover p-2 font-mono text-xs text-text-muted">
      {JSON.stringify(reasoning, null, 2)}
    </pre>
  );
}

function ReasoningRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 text-sm">
      <span className="text-text-muted">{label}</span>
      <span className="font-mono font-medium text-text">{value}</span>
    </div>
  );
}
