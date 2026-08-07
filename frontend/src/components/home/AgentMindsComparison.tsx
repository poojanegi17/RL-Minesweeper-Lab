import { useEffect, useState } from "react";
import { Trophy } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { EmptyState } from "@/components/ui/EmptyState";
import { ReplayControls } from "@/components/replay/ReplayControls";
import { RaceBoardTile } from "@/components/race/RaceBoardTile";
import { getRace, getRaces } from "@/api/races";
import { useApiQuery } from "@/hooks/useApiQuery";
import { agentKindFromName } from "@/lib/agentAdapters";

const SPEED_MS: Record<number, number> = { 1: 900, 2: 450, 4: 225 };

/**
 * "Different minds, same game" -- Random, CSP, DQN, and PPO all playing the
 * exact same seeded board, side by side, with real play/pause/speed
 * transport controls. Every episode here is pre-recorded (see
 * `rl/evaluation/generate_race.py`), not live inference -- the backend stays
 * a read-only static-artifact server. A single shared `stepIndex` drives all
 * four boards, so an agent that finishes early just holds its final
 * board/outcome while the others keep going -- a real competitive
 * comparison on identical mines, not four independently-sampled episodes.
 */
export function AgentMindsComparison() {
  const { data: races, status: listStatus, error: listError, isSlow: listSlow, retry: retryList } = useApiQuery(getRaces, []);

  const [raceId, setRaceId] = useState<string | null>(null);

  useEffect(() => {
    if (!races || races.length === 0) return;
    if (raceId === null || !races.some((r) => r.id === raceId)) {
      setRaceId(races[0].id);
    }
  }, [races, raceId]);

  const {
    data: race,
    status: raceStatus,
    error: raceError,
    isSlow: raceSlow,
    retry: retryRace,
  } = useApiQuery(() => (raceId ? getRace(raceId) : Promise.resolve(null)), [raceId]);

  const [stepIndex, setStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    setStepIndex(0);
    setIsPlaying(false);
  }, [raceId]);

  const agentEntries = race ? Object.entries(race.agents) : [];
  const totalSteps = agentEntries.length > 0 ? Math.max(...agentEntries.map(([, result]) => result.steps_taken)) : 0;

  useEffect(() => {
    if (!isPlaying || !race) return;
    if (stepIndex >= totalSteps) {
      setIsPlaying(false);
      return;
    }
    const timer = setTimeout(() => setStepIndex((i) => Math.min(i + 1, totalSteps)), SPEED_MS[speed]);
    return () => clearTimeout(timer);
  }, [isPlaying, stepIndex, totalSteps, speed, race]);

  return (
    <div className="flex flex-col gap-6">
      <p className="max-w-2xl text-sm text-text-muted">
        Same board, same mines, four different reasoning processes -- watch Random, CSP, DQN, and PPO each try to
        survive the identical layout.
      </p>

      {listStatus === "loading" && (
        <div className="flex flex-col gap-3" aria-label="Loading races">
          <Skeleton className="h-64 w-full" />
          {listSlow && <ColdStartNotice />}
        </div>
      )}

      {listStatus === "error" && listError && <ApiErrorState error={listError} onRetry={retryList} title="Couldn't load races" />}

      {listStatus === "success" && races && races.length === 0 && (
        <EmptyState
          icon={Trophy}
          title="No races generated yet"
          description="Run python -m evaluation.generate_race --episodes N to create some."
        />
      )}

      {listStatus === "success" && races && races.length > 0 && (
        <>
          {races.length > 1 && (
            <label className="flex w-full max-w-xs flex-col gap-1.5 text-xs text-text-muted">
              Board
              <Select value={raceId ?? ""} onChange={(e) => setRaceId(e.target.value)}>
                {races.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.id} (seed {r.seed})
                  </option>
                ))}
              </Select>
            </label>
          )}

          {raceStatus === "loading" && (
            <div className="flex flex-col gap-3">
              <Skeleton className="h-64 w-full" />
              {raceSlow && <ColdStartNotice />}
            </div>
          )}
          {raceStatus === "error" && raceError && (
            <ApiErrorState error={raceError} onRetry={retryRace} title="Couldn't load this board" />
          )}

          {raceStatus === "success" && race && (
            <div className="flex flex-col gap-4">
              <Card>
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

              <div className="grid gap-4 sm:grid-cols-2">
                {agentEntries.map(([agentName, result]) => (
                  <RaceBoardTile
                    key={agentName}
                    agentName={agentName}
                    kind={agentKindFromName(agentName)}
                    result={result}
                    initialBoard={race.initial_board}
                    stepIndex={stepIndex}
                  />
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
