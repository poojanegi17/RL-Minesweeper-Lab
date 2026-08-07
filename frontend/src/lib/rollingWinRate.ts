import type { MetricPoint } from "@/types/metrics";

export interface RollingWinRatePoint {
  episode: number;
  value: number;
}

/**
 * Rolling win rate over a trailing window of episodes, computed from `won`
 * -- present on every `MetricPoint` regardless of agent (unlike DQN's
 * precomputed `win_rate_rolling` field, which PPO's history doesn't carry).
 * Generic over agent for exactly that reason: it's the only way to compare
 * two different agents' learning curves on the same basis.
 *
 * `points` must already be episode-ordered (true of every history file this
 * project writes). Early episodes use a shrinking window (`min(i+1, window)`)
 * rather than padding with zeros, so the curve doesn't start at an
 * artificially low value.
 */
export function computeRollingWinRate(points: MetricPoint[], window: number): RollingWinRatePoint[] {
  if (points.length === 0 || window <= 0) return [];

  const result: RollingWinRatePoint[] = [];
  let wins = 0;

  for (let i = 0; i < points.length; i++) {
    if (points[i].won) wins++;
    if (i >= window && points[i - window].won) wins--;
    const count = Math.min(i + 1, window);
    result.push({ episode: points[i].episode, value: wins / count });
  }

  return result;
}
