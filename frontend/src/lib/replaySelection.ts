import type { ReplaySummary } from "@/types/replay";

/**
 * Picks the most representative replay from a list for the same agent --
 * prefers a win, then the longest episode, over plain array/filename order.
 * Shared by `Replay.tsx` (full episode browser) and `useAgentReplay` (every
 * "here's what {agent} actually did" preview) so a visitor never lands on an
 * agent's default replay purely because it happened to sort first
 * alphabetically (e.g. CSP's own `episode_1`, a 1-step loss, despite CSP
 * having real wins elsewhere in its replay set).
 */
export function pickBestReplay(replays: ReplaySummary[]): ReplaySummary | undefined {
  return [...replays].sort((a, b) => Number(b.won) - Number(a.won) || b.steps - a.steps)[0];
}
