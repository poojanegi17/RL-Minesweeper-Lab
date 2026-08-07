/** Builds the `?level=&density=` query string shared by every endpoint that
 * supports level/density scoping (`/api/leaderboard`, `/api/replays`,
 * `/api/races`). Omitting both (the default beginner/standard board) yields
 * an empty string -- identical to not passing the params at all, matching
 * the backend's own default. */
export function levelDensityQuery(level?: string, density?: string): string {
  const params = new URLSearchParams();
  if (level) params.set("level", level);
  if (density) params.set("density", density);
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}
