import { Trophy, Skull } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { AGENT_ICONS } from "@/components/agent/agentIcons";
import { AGENT_STYLES } from "@/data/types";
import { ReplayBoard } from "@/components/replay/ReplayBoard";
import { describeDecisionReason } from "@/lib/reasoning";
import { agentKindFromName } from "@/lib/agentAdapters";
import type { RaceDetail } from "@/types/race";
import { cn } from "@/lib/cn";

interface SharedRaceBoardProps {
  race: RaceDetail;
  /** 0 = the initial, all-hidden board; N = the board immediately after
   * `race.turns[N-1]`. Mirrors `Replay.tsx`'s `BoardAtStep` convention. */
  turnIndex: number;
}

/**
 * One physically shared board, driven by a single `turnIndex` -- not four
 * separate boards. Whoever's turn it is acts on the board every other agent
 * also sees; an eliminated agent's fatal cell never gets revealed to the
 * others (see `types/race.ts`), so the roster is the only place that
 * information shows up.
 */
export function SharedRaceBoard({ race, turnIndex }: SharedRaceBoardProps) {
  const currentTurn = turnIndex === 0 ? null : race.turns[turnIndex - 1];
  const board = currentTurn ? currentTurn.board_state : race.initial_board;
  const reason = currentTurn ? describeDecisionReason(currentTurn.agent, currentTurn.reasoning) : null;
  const isFinished = turnIndex >= race.total_turns;

  return (
    <div className="grid gap-4 lg:grid-cols-[auto_1fr]">
      <Card className="flex flex-col items-center gap-3">
        <ReplayBoard board={board} highlightedCell={currentTurn?.action ?? null} mineHit={currentTurn?.eliminated ?? false} />
        <p className="min-h-[2.5rem] text-center text-sm text-text-muted">
          {currentTurn ? (
            <>
              <span className={cn("font-medium", AGENT_STYLES[agentKindFromName(currentTurn.agent)].text)}>
                {currentTurn.agent}
              </span>
              {currentTurn.eliminated ? " hit a mine here -- eliminated." : reason ? ` -- ${reason}` : ""}
            </>
          ) : (
            "Waiting for the first move."
          )}
        </p>
      </Card>

      <Card className="flex flex-col gap-4">
        <div>
          <h3 className="mb-3 text-xs font-medium tracking-wide text-text-muted uppercase">Roster</h3>
          <ul className="flex flex-col gap-2">
            {race.turn_order.map((agentName) => {
              const style = AGENT_STYLES[agentKindFromName(agentName)];
              const Icon = AGENT_ICONS[agentKindFromName(agentName)];
              const eliminatedAtTurn = race.eliminated_agents[agentName];
              const isEliminated = eliminatedAtTurn !== undefined && turnIndex >= eliminatedAtTurn;
              const isUpNow = currentTurn?.agent === agentName && !isEliminated;

              return (
                <li key={agentName} className="flex items-center justify-between gap-2 text-sm">
                  <span className={cn("flex items-center gap-2", isEliminated ? "text-text-muted line-through" : style.text)}>
                    <Icon className="h-4 w-4" />
                    {agentName}
                  </span>
                  {isEliminated ? (
                    <Badge variant="outline">
                      <Skull className="h-3 w-3" />
                      Out (turn {eliminatedAtTurn})
                    </Badge>
                  ) : isUpNow ? (
                    <Badge variant="neutral">Up now</Badge>
                  ) : (
                    <Badge variant="outline">Alive</Badge>
                  )}
                </li>
              );
            })}
          </ul>
        </div>

        {isFinished && (
          <div
            className={cn(
              "flex items-center gap-2 rounded-lg border px-4 py-3 text-sm",
              race.won ? "border-emerald-500/30 bg-emerald-500/[0.08] text-emerald-700 dark:text-emerald-400" : "border-border bg-surface-hover/60 text-text-muted",
            )}
          >
            {race.won ? (
              <>
                <Trophy className="h-4 w-4 shrink-0" />
                Board cleared in {race.total_turns} turns -- survived by {race.surviving_agents.join(", ")}.
              </>
            ) : (
              <>
                <Skull className="h-4 w-4 shrink-0" />
                No winner -- every agent was eliminated after {race.total_turns} turns.
              </>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
