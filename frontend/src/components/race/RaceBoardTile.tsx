import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { AGENT_ICONS } from "@/components/agent/agentIcons";
import { AGENT_STYLES, type AgentKind } from "@/data/types";
import { ReplayBoard } from "@/components/replay/ReplayBoard";
import { describeDecisionReason } from "@/lib/reasoning";
import type { RaceAgentResult } from "@/types/race";
import { cn } from "@/lib/cn";

interface RaceBoardTileProps {
  agentName: string;
  kind: AgentKind;
  result: RaceAgentResult;
  initialBoard: number[][];
  /** The race's shared step index -- every tile is driven by the same value,
   * not its own. An agent whose episode ended earlier than others just holds
   * its final board/outcome for any step beyond its own `steps_taken`
   * (natural "race" semantics: it finished, the others are still going). */
  stepIndex: number;
}

export function RaceBoardTile({ agentName, kind, result, initialBoard, stepIndex }: RaceBoardTileProps) {
  const style = AGENT_STYLES[kind];
  const Icon = AGENT_ICONS[kind];

  const isFinished = stepIndex >= result.steps_taken;
  const localStepIndex = Math.min(stepIndex, result.steps_taken);
  const step = localStepIndex === 0 ? null : result.steps[localStepIndex - 1];
  const board = step ? step.board_state : initialBoard;
  const mineHit = isFinished && !result.won;
  const reason = step ? describeDecisionReason(agentName, step.reasoning) : null;

  return (
    <Card className="flex flex-col items-center gap-3">
      <div className="flex w-full items-center justify-between">
        <div className={cn("flex items-center gap-2", style.text)}>
          <Icon className="h-4 w-4" />
          <h3 className="text-sm font-semibold text-heading">{agentName}</h3>
        </div>
        <Badge
          variant={isFinished ? "neutral" : "outline"}
          className={cn(isFinished && result.won && "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400")}
        >
          {isFinished ? (result.won ? "WIN" : "LOSS") : "In progress"}
        </Badge>
      </div>

      <ReplayBoard board={board} highlightedCell={step?.action ?? null} mineHit={mineHit} />

      <p className="min-h-[2.5rem] text-center text-xs text-text-muted">
        {step ? reason ?? "No reasoning recorded for this step." : "Waiting for the first move."}
      </p>
    </Card>
  );
}
