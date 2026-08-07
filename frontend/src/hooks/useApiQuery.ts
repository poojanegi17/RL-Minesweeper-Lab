import { useCallback, useEffect, useState, type DependencyList } from "react";

export type QueryStatus = "loading" | "success" | "error";

export interface UseApiQueryResult<T> {
  data: T | null;
  status: QueryStatus;
  error: Error | null;
  /** True once the current request has been loading for longer than
   * `SLOW_THRESHOLD_MS` -- distinguishes a normal brief loading flicker from
   * a Render free-tier cold start (can take 20-50s), so pages can show a
   * "waking up the backend" hint instead of leaving a bare skeleton up with
   * no explanation. Always false outside the "loading" status. */
  isSlow: boolean;
  /** Re-runs `fetcher` from scratch -- wired to the "Retry" action in error states. */
  retry: () => void;
}

const SLOW_THRESHOLD_MS = 3500;

/**
 * Small `useState`/`useEffect`-based data-fetching hook -- this project has
 * no React Query/SWR dependency, and "connect the frontend to the backend"
 * doesn't warrant adding one. Every page that calls an `src/api/*` function
 * goes through this hook so loading/error/retry behavior is consistent.
 *
 * `fetcher` is intentionally excluded from the effect's dependency array
 * (inline arrow functions are re-created every render, which would loop
 * forever) -- pass whatever `fetcher` actually depends on via `deps`
 * instead, e.g. `useApiQuery(() => getExperiment(id), [id])`.
 */
export function useApiQuery<T>(fetcher: () => Promise<T>, deps: DependencyList = []): UseApiQueryResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [status, setStatus] = useState<QueryStatus>("loading");
  const [error, setError] = useState<Error | null>(null);
  const [isSlow, setIsSlow] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    setError(null);
    setIsSlow(false);

    const slowTimer = setTimeout(() => {
      if (!cancelled) setIsSlow(true);
    }, SLOW_THRESHOLD_MS);

    fetcher()
      .then((result) => {
        if (cancelled) return;
        setData(result);
        setStatus("success");
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err : new Error("Something went wrong."));
        setStatus("error");
      })
      .finally(() => {
        clearTimeout(slowTimer);
      });

    return () => {
      cancelled = true;
      clearTimeout(slowTimer);
    };
    // deps is caller-controlled by design -- see docstring.
    // eslint-disable-next-line
  }, [...deps, attempt]);

  const retry = useCallback(() => setAttempt((n) => n + 1), []);

  return { data, status, error, isSlow, retry };
}
