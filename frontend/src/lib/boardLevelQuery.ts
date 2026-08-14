/** Builds the `?level=&density=` query string shared by every endpoint that
 * supports level/density scoping (`/api/leaderboard`, `/api/replays`,
 * `/api/races`). Omitting both (the default beginner/standard board) yields
 * an empty string -- identical to not passing the params at all, matching
 * the backend's own default. */
export function levelDensityQuery(level?: string, density?: string, firstClickSafe?: FirstClickPolicy): string {
  const params = new URLSearchParams();
  if (level) params.set("level", level);
  if (density) params.set("density", density);
  if (firstClickSafe) params.set("first_click_safe", firstClickSafe);
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

/**
 * The two board distributions this project has published grids under.
 * `"area"` keeps the whole 3x3 block around the opening click mine-free, so a
 * game can never be lost on move one; `"none"` places mines before the first
 * click. They are different games -- a win rate under one is not comparable to
 * a win rate under the other, which is why the UI switches between them rather
 * than showing both on one axis.
 *
 * Only `/api/leaderboard` accepts this today; passing it selects that
 * distribution's own results tree instead of the default one.
 */
export type FirstClickPolicy = "area" | "none";

export const FIRST_CLICK_POLICY_LABELS: Record<FirstClickPolicy, string> = {
  area: "First click safe",
  none: "First click unsafe",
};

export const FIRST_CLICK_POLICY_BLURBS: Record<FirstClickPolicy, string> = {
  area: "The opening click always opens a mine-free 3×3 block, so every game starts with a cascade and no game is lost on move one.",
  none: "Mines are placed before the first click, so the opening move can hit one — on a 5×5 board with 5 mines that alone loses about a fifth of all games.",
};
