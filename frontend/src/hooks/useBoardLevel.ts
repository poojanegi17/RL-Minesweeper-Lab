import { useState } from "react";
import { getBoardConfigs } from "@/api/boardConfigs";
import { useApiQuery } from "@/hooks/useApiQuery";

/**
 * Fetches the real board-config catalog (`GET /api/board-configs`) and holds
 * the locally-selected `(level, density)` -- shared by every surface with a
 * `LevelDensitySelector` (`AIComparisonBoard`, `Replay`, `AgentMindsComparison`,
 * `Research`, `PlayableMinesweeper`), each with its own independent
 * selection, not a single global one (these are genuinely separate tools).
 */
export function useBoardLevel(defaultLevel = "beginner", defaultDensity = "standard") {
  const { data: configs, status, error, isSlow, retry } = useApiQuery(getBoardConfigs, []);
  const [level, setLevel] = useState(defaultLevel);
  const [density, setDensity] = useState(defaultDensity);

  return {
    configs: configs ?? [],
    status,
    error,
    isSlow,
    retry,
    level,
    density,
    setLevel,
    setDensity,
  };
}
