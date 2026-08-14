import { ArrowLeftRight } from "lucide-react";
import { Select } from "@/components/ui/Select";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { CompareStatRow } from "@/components/compare/CompareStatRow";
import { LevelDensitySelector } from "@/components/board/LevelDensitySelector";
import { getBoardConfigs } from "@/api/boardConfigs";
import { getLeaderboard } from "@/api/metrics";
import { useApiQuery } from "@/hooks/useApiQuery";
import { agentKindFromName } from "@/lib/agentAdapters";
import { type FirstClickPolicy } from "@/lib/boardLevelQuery";

interface EnvironmentCompareCardProps {
  /** Which board distribution to read, set by the page's environment toggle. */
  policy: FirstClickPolicy;
  /** Board size and density. Owned by the page for the same reason `nameA`/
   * `nameB` are: the training-run half below reads the board size too, to
   * resolve which run to describe. Held here previously, which is exactly why
   * that half could not follow the selection. */
  level: string;
  density: string;
  onLevelChange: (level: string) => void;
  onDensityChange: (density: string) => void;
  /** Which agents may be picked -- the shared catalog, so the list stays the
   * same across environments even where one has no result for an agent. */
  agentNames: string[];
  /** Selection is owned by the page, because the same pair drives both this
   * board-cell comparison and the training-run detail beneath it. */
  nameA: string;
  nameB: string;
  onChangeA: (name: string) => void;
  onChangeB: (name: string) => void;
}

/**
 * The side-by-side comparison for whichever environment is currently selected.
 *
 * Board size and mine density are chosen here because they vary *within* one
 * distribution, and every agent is re-measured across them. The environment
 * itself sits one level up, in `EnvironmentToggle`, because changing it
 * changes the game rather than the configuration -- and it is placed above
 * these controls so the page reads environment, then board, then agents.
 *
 * Only headline stats appear here. Learning curves and hyperparameters come
 * from training artifacts, which belong to a run rather than to a board cell,
 * so they stay on the page's lower half where they are labelled as such.
 */
export function EnvironmentCompareCard({
  policy,
  level,
  density,
  onLevelChange,
  onDensityChange,
  agentNames,
  nameA,
  nameB,
  onChangeA,
  onChangeB,
}: EnvironmentCompareCardProps) {
  const { data, status, error, isSlow, retry } = useApiQuery(async () => {
    const [configs, leaderboard] = await Promise.all([getBoardConfigs(), getLeaderboard(level, density, policy)]);
    return { configs, leaderboard };
  }, [level, density, policy]);

  const entryFor = (name: string) => data?.leaderboard.find((entry) => entry.agent === name);

  return (
    <div className="flex flex-col gap-4">
      {status === "loading" && (
        <div className="flex flex-col gap-3">
          <Skeleton className="h-40 w-full" />
          {isSlow && <ColdStartNotice />}
        </div>
      )}

      {status === "error" && error && (
        <ApiErrorState error={error} onRetry={retry} title="Couldn't load this environment's results" />
      )}

      {status === "success" && data && (
        <>
          <LevelDensitySelector
            configs={data.configs}
            level={level}
            density={density}
            onLevelChange={onLevelChange}
            onDensityChange={onDensityChange}
          />

          <div className="flex flex-col items-center gap-3 sm:flex-row">
            <Select className="sm:max-w-xs" aria-label="Agent A" value={nameA} onChange={(e) => onChangeA(e.target.value)}>
              {agentNames
                .filter((name) => name !== nameB)
                .map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
            </Select>

            <Button
              variant="secondary"
              size="sm"
              aria-label="Swap agents"
              onClick={() => {
                const previousA = nameA;
                onChangeA(nameB);
                onChangeB(previousA);
              }}
            >
              <ArrowLeftRight className="h-3.5 w-3.5" />
            </Button>

            <Select className="sm:max-w-xs" aria-label="Agent B" value={nameB} onChange={(e) => onChangeB(e.target.value)}>
              {agentNames
                .filter((name) => name !== nameA)
                .map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
            </Select>
          </div>

          <CompareStatRow
            agentA={{ name: nameA, kind: agentKindFromName(nameA), leaderboardEntry: entryFor(nameA) }}
            agentB={{ name: nameB, kind: agentKindFromName(nameB), leaderboardEntry: entryFor(nameB) }}
          />

          <p className="text-xs text-text-muted">
            2,000 evaluation episodes per agent, seed 42 — every agent faces the identical boards at this cell, so a
            difference between them is a difference in play rather than in luck of the draw.
          </p>
        </>
      )}
    </div>
  );
}
