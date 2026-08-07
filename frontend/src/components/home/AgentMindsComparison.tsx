import { useEffect, useState } from "react";
import { Trophy } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { EmptyState } from "@/components/ui/EmptyState";
import { ReplayControls } from "@/components/replay/ReplayControls";
import { SharedRaceBoard } from "@/components/race/SharedRaceBoard";
import { getRace, getRaces } from "@/api/races";
import { useApiQuery } from "@/hooks/useApiQuery";

const SPEED_MS: Record<number, number> = { 1: 900, 2: 450, 4: 225 };

/**
 * "Different minds, same game" -- Random, CSP, DQN, and PPO take turns on
 * one physically shared board, with real play/pause/speed transport
 * controls. Every race is pre-recorded (see `rl/evaluation/generate_race.py`),
 * not live inference -- the backend stays a read-only static-artifact
 * server. Whatever's revealed stays revealed for everyone; an agent that
 * hits a mine is eliminated but the board keeps going for whoever's left --
 * see `SharedRaceBoard` for why this is a meaningfully different thing from
 * four independently-sampled episodes on matching seeds.
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

  const totalTurns = race?.total_turns ?? 0;

  useEffect(() => {
    if (!isPlaying || !race) return;
    if (stepIndex >= totalTurns) {
      setIsPlaying(false);
      return;
    }
    const timer = setTimeout(() => setStepIndex((i) => Math.min(i + 1, totalTurns)), SPEED_MS[speed]);
    return () => clearTimeout(timer);
  }, [isPlaying, stepIndex, totalTurns, speed, race]);

  return (
    <div className="flex flex-col gap-6">
      <p className="max-w-2xl text-sm text-text-muted">
        One shared board, four agents taking turns -- Random, CSP, DQN, and PPO. Whatever gets revealed stays
        revealed for everyone; a mistake only costs the agent who made it.
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
                  totalSteps={totalTurns}
                  isPlaying={isPlaying}
                  speed={speed}
                  onPrevious={() => setStepIndex((i) => Math.max(0, i - 1))}
                  onNext={() => setStepIndex((i) => Math.min(totalTurns, i + 1))}
                  onTogglePlay={() => setIsPlaying((p) => !p)}
                  onSpeedChange={setSpeed}
                />
              </Card>

              <SharedRaceBoard race={race} turnIndex={stepIndex} />
            </div>
          )}
        </>
      )}
    </div>
  );
}
