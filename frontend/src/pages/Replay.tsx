import { useEffect, useMemo, useState } from "react";
import { Film } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { Skeleton } from "@/components/ui/Skeleton";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { EmptyState } from "@/components/ui/EmptyState";
import { ReplayBoard } from "@/components/replay/ReplayBoard";
import { ReplayControls } from "@/components/replay/ReplayControls";
import { ReplayTimeline } from "@/components/replay/ReplayTimeline";
import { ReplayInfo } from "@/components/replay/ReplayInfo";
import { getReplay, getReplays } from "@/api/replays";
import { useApiQuery } from "@/hooks/useApiQuery";
import type { ReplayDetail } from "@/types/replay";

const SPEED_MS: Record<number, number> = { 1: 900, 2: 450, 4: 225 };

export function Replay() {
  const { data: replays, status: listStatus, error: listError, retry: retryList } = useApiQuery(getReplays, []);

  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [selectedReplayId, setSelectedReplayId] = useState<string | null>(null);

  const agentOptions = useMemo(() => Array.from(new Set((replays ?? []).map((r) => r.agent))).sort(), [replays]);
  const replaysForAgent = useMemo(
    () => (replays ?? []).filter((r) => r.agent === (selectedAgent ?? agentOptions[0])),
    [replays, selectedAgent, agentOptions],
  );

  // Default to the first agent/replay once the list has loaded.
  useEffect(() => {
    if (!replays || replays.length === 0) return;
    if (selectedAgent === null) {
      setSelectedAgent(agentOptions[0] ?? null);
    }
  }, [replays, agentOptions, selectedAgent]);

  useEffect(() => {
    if (replaysForAgent.length === 0) {
      setSelectedReplayId(null);
      return;
    }
    if (!replaysForAgent.some((r) => r.id === selectedReplayId)) {
      setSelectedReplayId(replaysForAgent[0].id);
    }
  }, [replaysForAgent, selectedReplayId]);

  const {
    data: replay,
    status: detailStatus,
    error: detailError,
    retry: retryDetail,
  } = useApiQuery(() => (selectedReplayId ? getReplay(selectedReplayId) : Promise.resolve(null)), [selectedReplayId]);

  const [stepIndex, setStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  // Reset playback whenever a different replay is selected.
  useEffect(() => {
    setStepIndex(0);
    setIsPlaying(false);
  }, [selectedReplayId]);

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

  return (
    <div className="flex flex-col gap-10">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight text-heading">
          Replay Viewer
        </h1>
        <p className="mt-2 max-w-2xl text-text-muted">
          Watch how each agent actually plays, move by move — only what the
          agent itself could see at the time, never the mine layout ahead of
          when it was discovered.
        </p>
      </div>

      {listStatus === "loading" && (
        <div className="flex flex-col gap-4" aria-label="Loading replays">
          <Skeleton className="h-10 w-full max-w-md" />
          <Skeleton className="h-96 w-full" />
        </div>
      )}

      {listStatus === "error" && listError && (
        <ApiErrorState error={listError} onRetry={retryList} title="Couldn't load replays" />
      )}

      {listStatus === "success" && replays && replays.length === 0 && (
        <EmptyState
          icon={Film}
          title="No replays generated yet"
          description="Run python -m evaluation.generate_replays --agent <csp|dqn|ppo|random> --episodes N to create some."
        />
      )}

      {listStatus === "success" && replays && replays.length > 0 && (
        <>
          <div className="flex flex-col items-center gap-4 sm:flex-row">
            <label className="flex w-full flex-col gap-1.5 text-sm text-text-muted sm:max-w-xs">
              Agent
              <Select value={selectedAgent ?? ""} onChange={(e) => setSelectedAgent(e.target.value)}>
                {agentOptions.map((agent) => (
                  <option key={agent} value={agent}>
                    {agent}
                  </option>
                ))}
              </Select>
            </label>
            <label className="flex w-full flex-col gap-1.5 text-sm text-text-muted sm:max-w-xs">
              Episode
              <Select value={selectedReplayId ?? ""} onChange={(e) => setSelectedReplayId(e.target.value)}>
                {replaysForAgent.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.id} ({r.won ? "WIN" : "loss"}, {r.steps} steps)
                  </option>
                ))}
              </Select>
            </label>
          </div>

          {detailStatus === "loading" && <Skeleton className="h-96 w-full" />}
          {detailStatus === "error" && detailError && (
            <ApiErrorState error={detailError} onRetry={retryDetail} title="Couldn't load this replay" />
          )}
          {detailStatus === "success" && replay && (
            <div className="grid gap-6 lg:grid-cols-[auto_1fr]">
              <Card className="flex flex-col gap-4">
                <ReplayTimeline stepIndex={stepIndex} totalSteps={totalSteps} onScrub={setStepIndex} />
                <BoardAtStep replay={replay} stepIndex={stepIndex} />
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
              </Card>

              <Card>
                <ReplayInfo replay={replay} currentStep={stepIndex === 0 ? null : replay.timeline[stepIndex - 1]} />
              </Card>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function BoardAtStep({ replay, stepIndex }: { replay: ReplayDetail; stepIndex: number }) {
  const step = stepIndex === 0 ? null : replay.timeline[stepIndex - 1];
  const board = step ? step.board_state : replay.initial_board;
  const isLastStep = stepIndex === replay.steps;
  const mineHit = isLastStep && !replay.won;

  return <ReplayBoard board={board} highlightedCell={step?.action ?? null} mineHit={mineHit} />;
}
