import { useCallback, useEffect, useState, type DependencyList } from "react";

export type QueryStatus = "loading" | "success" | "error";

export interface UseApiQueryResult<T> {
  data: T | null;
  status: QueryStatus;
  error: Error | null;
  /** Re-runs `fetcher` from scratch -- wired to the "Retry" action in error states. */
  retry: () => void;
}

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
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    setError(null);

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
      });

    return () => {
      cancelled = true;
    };
    // deps is caller-controlled by design -- see docstring.
    // eslint-disable-next-line
  }, [...deps, attempt]);

  const retry = useCallback(() => setAttempt((n) => n + 1), []);

  return { data, status, error, retry };
}
