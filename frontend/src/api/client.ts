/**
 * Thin fetch wrapper shared by every module in `src/api/`. Nothing outside
 * this directory should call `fetch` directly or read `VITE_API_URL` --
 * components go through `agents.ts`/`experiments.ts`/`metrics.ts` instead,
 * so the base URL and error handling live in exactly one place.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL;
const API_BASE_URL_IS_SET = Boolean(API_BASE_URL);

/** Thrown for both non-2xx responses and network-level failures (backend
 * offline, DNS failure, etc.) -- `status` is `null` for the latter, so
 * callers can distinguish "server said no" from "couldn't reach the server"
 * without a separate exception type. */
export class ApiError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  if (!API_BASE_URL_IS_SET) {
    throw new ApiError(
      "VITE_API_URL is not set -- the app doesn't know which backend to call. " +
        "Set it in frontend/.env (local) or as a project environment variable (deployed).",
      null,
    );
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`);
  } catch {
    throw new ApiError(
      "Could not reach the backend API. Is it running at " + API_BASE_URL + "?",
      null,
    );
  }

  if (!response.ok) {
    const detail = await response
      .json()
      .then((body: { detail?: string }) => body.detail)
      .catch(() => undefined);
    throw new ApiError(detail ?? `Request to ${path} failed (${response.status})`, response.status);
  }

  return response.json() as Promise<T>;
}
