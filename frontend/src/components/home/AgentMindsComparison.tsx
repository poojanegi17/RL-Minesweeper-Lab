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
import { LevelDensitySelector } from "@/components/board/LevelDensitySelector";
import { getRace, getRaces } from "@/api/races";
import { useApiQuery } from "@/hooks/useApiQuery";
import { useBoardLevel } from "@/hooks/useBoardLevel";
import { FIRST_CLICK_POLICY_LABELS, type FirstClickPolicy } from "@/lib/boardLevelQuery";
import { cn } from "@/lib/cn";

const SPEED_MS: Record<number, number> = { 1: 900, 2: 450, 4: 225 };

const POLICIES: FirstClickPolicy[] = ["none", "area"];

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
  const { configs, level, density, setLevel, setDensity } = useBoardLevel();
  // Defaults to "none" to match the rest of the site's default view. The two
  // are different games, so races are never mixed across the toggle -- each
  // policy reads its own tree.
  const [policy, setPolicy] = useState<FirstClickPolicy>("none");
  const {
    data: races,
    status: listStatus,
    error: listError,
    isSlow: listSlow,
    retry: retryList,
  } = useApiQuery(() => getRaces(level, density, policy), [level, density, policy]);

  const [raceId, setRaceId] = useState<string | null>(null);

  useEffect(() => {
    if (!races || races.length === 0) {
      setRaceId(null);
      return;
    }
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
  } = useApiQuery(
    () => (raceId ? getRace(raceId, level, density, policy) : Promise.resolve(null)),
    [raceId, level, density, policy],
  );

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

      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="inline-flex rounded-xl border border-border p-1" role="group" aria-label="Board distribution">
          {POLICIES.map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => setPolicy(option)}
              aria-pressed={policy === option}
              className={cn(
                "rounded-lg px-3.5 py-1.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60",
                policy === option ? "bg-primary text-white shadow-sm" : "text-text-muted hover:text-text",
              )}
            >
              {FIRST_CLICK_POLICY_LABELS[option]}
            </button>
          ))}
        </div>
        <p className="text-xs text-text-muted">
          Under an unsafe opening the first agent to move can lose on turn one through no fault of its policy.
        </p>
      </div>

      {configs.length > 0 && (
        <LevelDensitySelector configs={configs} level={level} density={density} onLevelChange={setLevel} onDensityChange={setDensity} compact />
      )}

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
          title="No races generated at this level yet"
          description={
            level === "beginner" && density === "standard"
              ? "Run python -m evaluation.generate_race --episodes N to create some."
              : "Races need all four agents trained at this level -- DQN/PPO haven't been trained here yet."
          }
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
