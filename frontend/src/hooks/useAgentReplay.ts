import { useMemo } from "react";
import { getReplay, getReplays } from "@/api/replays";
import { useApiQuery, type QueryStatus } from "@/hooks/useApiQuery";
import type { ReplayDetail } from "@/types/replay";

export interface UseAgentReplayResult {
  /** The first recorded replay for `agentName`, or null while loading/on
   * error/if none exists. */
  replay: ReplayDetail | null;
  /** "success" only once both the replay list and the selected replay's
   * detail have loaded -- never "success" with a stale `replay` from a
   * previous `agentName`. */
  status: QueryStatus;
  error: Error | null;
  /** True once either underlying request has been loading long enough to
   * look like a backend cold start rather than a normal brief flicker --
   * see `useApiQuery`'s `isSlow`. */
  isSlow: boolean;
  retry: () => void;
}

/**
 * Fetches one representative recorded episode for a given agent: `GET
 * /api/replays` (list), then `GET /api/replays/{id}` (detail) for the first
 * match. Shared by every UI that shows "here's what {agent} actually did" --
 * `DecisionExample` (agent explainer pages) and `AIComparisonBoard` (home
 * page) both need exactly this chain, so it lives here once instead of
 * twice.
 */
export function useAgentReplay(agentName: string | null): UseAgentReplayResult {
  const {
    data: replays,
    status: listStatus,
    error: listError,
    isSlow: listSlow,
    retry: retryList,
  } = useApiQuery(getReplays, []);
  const replayId = useMemo(
    () => (agentName ? (replays?.find((r) => r.agent === agentName)?.id ?? null) : null),
    [replays, agentName],
  );

  const {
    data: replay,
    status: detailStatus,
    error: detailError,
    isSlow: detailSlow,
    retry: retryDetail,
  } = useApiQuery(() => (replayId ? getReplay(replayId) : Promise.resolve(null)), [replayId]);

  const status: QueryStatus =
    listStatus === "error" || detailStatus === "error"
      ? "error"
      : listStatus === "loading" || (replayId !== null && detailStatus === "loading")
        ? "loading"
        : "success";

  return {
    replay: replayId ? replay : null,
    status,
    error: listError ?? detailError,
    isSlow: listSlow || detailSlow,
    retry: () => {
      retryList();
      retryDetail();
    },
  };
}
