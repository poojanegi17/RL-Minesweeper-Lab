import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { PlayCircle, User } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { EmptyState } from "@/components/ui/EmptyState";
import { ReplayBoard } from "@/components/replay/ReplayBoard";
import { ReplayControls } from "@/components/replay/ReplayControls";
import { ReplayTimeline } from "@/components/replay/ReplayTimeline";
import { LevelDensitySelector } from "@/components/board/LevelDensitySelector";
import { AGENT_ICONS } from "@/components/agent/agentIcons";
import { AGENT_STYLES } from "@/data/types";
import { agentKindFromName } from "@/lib/agentAdapters";
import { useAgentReplay } from "@/hooks/useAgentReplay";
import { useBoardLevel } from "@/hooks/useBoardLevel";
import { describeDecisionReason } from "@/lib/reasoning";
import { cn } from "@/lib/cn";
import type { PlayableMinesweeperSummary } from "@/components/home/PlayableMinesweeper";

const AGENT_OPTIONS = [
  { label: "CSP Solver", value: "CSP" },
  { label: "Q-Learning", value: "Q-Learning" },
  { label: "DQN", value: "DQN" },
  { label: "PPO", value: "PPO" },
  { label: "Random", value: "Random" },
];

const SPEED_MS: Record<number, number> = { 1: 900, 2: 450, 4: 225 };

interface AIComparisonBoardProps {
  /** The visitor's own progress on `PlayableMinesweeper` (Section 2, right
   * above this one) -- purely a readout, this component never touches that
   * game's state directly. */
  humanSummary: PlayableMinesweeperSummary;
}

/**
 * "Now watch AI play" -- pick any agent and step through (or autoplay) a
 * real recorded episode of it, right next to a readout of the visitor's own
 * game. Built entirely from reused pieces: `useAgentReplay` and the Replay
 * Viewer's own `ReplayBoard`/`ReplayControls`/`ReplayTimeline` -- no parallel
 * replay logic here.
 */
export function AIComparisonBoard({ humanSummary }: AIComparisonBoardProps) {
  const [agent, setAgent] = useState("DQN");
  const { configs, level, density, setLevel, setDensity } = useBoardLevel();
  const { replay, status, error, isSlow, retry } = useAgentReplay(agent, level, density);

  const [stepIndex, setStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    setStepIndex(0);
    setIsPlaying(false);
  }, [replay?.id]);

  const totalSteps = replay?.steps ?? 0;

  useEffect(() => {
    if (!isPlaying || !replay) return;
    if (stepIndex >= totalSteps) {
      setIsPlaying(false);
      return;
    }
    const timer = setTimeout(() => setStepIndex((i) => Math.min(i + 1, totalSteps)), SPEED_MS[speed]);
    return () => clearTimeout(timer);
  }, [isPlaying, stepIndex, totalSteps, speed, replay]);

  const step = replay && stepIndex > 0 ? replay.timeline[stepIndex - 1] : null;
  const board = step ? step.board_state : (replay?.initial_board ?? null);
  const isLastStep = replay ? stepIndex === replay.steps : false;
  const mineHit = isLastStep && replay ? !replay.won : false;
  const reason = step && replay ? describeDecisionReason(replay.agent, step.reasoning) : null;

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,220px)_1fr]">
      <Card className="flex flex-col gap-3">
        <div className="flex items-center gap-2 text-text-muted">
          <User className="h-4 w-4" />
          <h3 className="text-xs font-medium tracking-wide uppercase">Your strategy</h3>
        </div>
        <dl className="flex flex-col gap-2 text-sm">
          <div className="flex items-center justify-between">
            <dt className="text-text-muted">Status</dt>
            <dd className="font-medium text-heading capitalize">
              {humanSummary.status === "idle" ? "Not started" : humanSummary.status}
            </dd>
          </div>
          <div className="flex items-center justify-between">
            <dt className="text-text-muted">Cells revealed</dt>
            <dd className="font-mono font-medium text-heading">{humanSummary.revealedCount}</dd>
          </div>
          <div className="flex items-center justify-between">
            <dt className="text-text-muted">Flags placed</dt>
            <dd className="font-mono font-medium text-heading">{humanSummary.flagCount}</dd>
          </div>
        </dl>
        <p className="text-xs text-text-muted">Play the board above, then compare your choices to an agent's.</p>
      </Card>

      <Card className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className={cn("flex items-center gap-2", AGENT_STYLES[agentKindFromName(agent)].text)}>
            <AgentIcon agent={agent} />
            <h3 className="text-xs font-medium tracking-wide text-text-muted uppercase">Agent strategy</h3>
          </div>
          <div className="w-40">
            <Select value={agent} onChange={(e) => setAgent(e.target.value)} aria-label="Select agent">
              {AGENT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </Select>
          </div>
        </div>

        {configs.length > 0 && (
          <LevelDensitySelector configs={configs} level={level} density={density} onLevelChange={setLevel} onDensityChange={setDensity} compact />
        )}

        {status === "loading" && (
          <div className="flex flex-col gap-3">
            <Skeleton className="h-64 w-full" />
            {isSlow && <ColdStartNotice />}
          </div>
        )}
        {status === "error" && error && (
          <ApiErrorState error={error} onRetry={retry} title="Couldn't load this agent's replay" />
        )}
        {status === "success" && !replay && (
          <EmptyState
            icon={PlayCircle}
            title="No recorded episodes yet"
            description={
              level === "beginner" && density === "standard"
                ? `No replay has been generated for ${agent} yet.`
                : `${agent} hasn't been trained/recorded at this level and density yet.`
            }
          />
        )}
        <AnimatePresence mode="wait">
          {status === "success" && replay && board && (
            <motion.div
              key={agent}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25, ease: "easeOut" }}
              className="flex flex-col gap-4 sm:flex-row sm:items-start"
            >
              <div className="flex flex-col items-center gap-3">
                <ReplayBoard board={board} highlightedCell={step?.action ?? null} mineHit={mineHit} />
                <ReplayTimeline stepIndex={stepIndex} totalSteps={totalSteps} onScrub={setStepIndex} />
                <ReplayControls
                  stepIndex={stepIndex}
                  totalSteps={totalSteps}
                  isPlaying={isPlaying}
                  speed={speed}
                  onPrevious={() => setStepIndex((i) => Math.max(0, i - 1))}
                  onNext={() => setStepIndex((i) => Math.min(totalSteps, i + 1))}
                  onTogglePlay={() => setIsPlaying((p) => !p)}
                  onSpeedChange={setSpeed}
                />
              </div>
              <div className="flex flex-1 flex-col gap-2">
                <Badge variant="outline" className="w-fit">
                  {replay.won ? "Win" : "Loss"} · {replay.steps} steps
                </Badge>
                {step ? (
                  <>
                    <p className="text-sm text-text">
                      Selected cell: <span className="font-mono">({step.action.row}, {step.action.col})</span>
                    </p>
                    <p className="text-sm text-text-muted">{reason ? `Reason: ${reason}` : "No reasoning recorded for this step."}</p>
                  </>
                ) : (
                  <p className="text-sm text-text-muted">Press Play (or step forward) to watch {agent} decide.</p>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>
    </div>
  );
}

/** The selected agent's own catalog icon, tinted with its own identity
 * color -- reused lookups (`agentKindFromName`/`AGENT_ICONS`), not a new icon set. */
function AgentIcon({ agent }: { agent: string }) {
  const Icon = AGENT_ICONS[agentKindFromName(agent)];
  return <Icon className="h-4 w-4" />;
}
