import { Card } from "@/components/ui/Card";
import { AGENT_ICONS } from "@/components/agent/agentIcons";
import { AGENT_STYLES, type AgentKind } from "@/data/types";
import type { LeaderboardEntry } from "@/types/metrics";
import { cn } from "@/lib/cn";

interface CompareAgent {
  name: string;
  kind: AgentKind;
  leaderboardEntry: LeaderboardEntry | undefined;
}

interface CompareStatRowProps {
  agentA: CompareAgent;
  agentB: CompareAgent;
}

interface StatRow {
  label: string;
  format: (entry: LeaderboardEntry | undefined) => string;
  /** Which side has the "better" value, for a subtle highlight -- only for
   * metrics where higher is unambiguously better. Avg. episode length isn't
   * included: a longer episode isn't clearly better or worse (a strong agent
   * dying in 2 steps only means a genuinely unlucky first move). */
  better?: (entry: LeaderboardEntry | undefined) => number | null;
}

const ROWS: StatRow[] = [
  {
    label: "Win rate",
    format: (entry) => (entry?.win_rate != null ? `${(entry.win_rate * 100).toFixed(1)}%` : "—"),
    better: (entry) => entry?.win_rate ?? null,
  },
  {
    label: "Avg. reward",
    format: (entry) => (entry?.avg_reward != null ? entry.avg_reward.toFixed(2) : "—"),
    better: (entry) => entry?.avg_reward ?? null,
  },
  {
    label: "Avg. episode length",
    format: (entry) => (entry?.avg_episode_length != null ? entry.avg_episode_length.toFixed(2) : "—"),
  },
];

/** Headline win-rate/reward/episode-length stats for two agents, side by
 * side, with the better value (where "better" is unambiguous) subtly
 * highlighted -- doubled version of `AgentOverview.tsx`'s win-rate stat card. */
export function CompareStatRow({ agentA, agentB }: CompareStatRowProps) {
  return (
    <Card className="flex flex-col gap-5">
      <div className="grid grid-cols-2 gap-4">
        <AgentHeader name={agentA.name} kind={agentA.kind} Icon={AGENT_ICONS[agentA.kind]} />
        <AgentHeader name={agentB.name} kind={agentB.kind} Icon={AGENT_ICONS[agentB.kind]} align="right" />
      </div>

      <dl className="flex flex-col divide-y divide-border">
        {ROWS.map((row) => {
          const valueA = row.better?.(agentA.leaderboardEntry) ?? null;
          const valueB = row.better?.(agentB.leaderboardEntry) ?? null;
          const aIsBetter = valueA != null && valueB != null && valueA > valueB;
          const bIsBetter = valueA != null && valueB != null && valueB > valueA;

          return (
            <div key={row.label} className="grid grid-cols-[1fr_auto_1fr] items-center gap-3 py-3">
              <dd className={cn("font-mono text-sm font-semibold text-heading", aIsBetter && "text-emerald-500")}>
                {row.format(agentA.leaderboardEntry)}
              </dd>
              <dt className="text-center text-xs text-text-muted uppercase">{row.label}</dt>
              <dd className={cn("text-right font-mono text-sm font-semibold text-heading", bIsBetter && "text-emerald-500")}>
                {row.format(agentB.leaderboardEntry)}
              </dd>
            </div>
          );
        })}
      </dl>
    </Card>
  );
}

function AgentHeader({
  name,
  kind,
  Icon,
  align = "left",
}: {
  name: string;
  kind: AgentKind;
  Icon: (typeof AGENT_ICONS)[AgentKind];
  align?: "left" | "right";
}) {
  const style = AGENT_STYLES[kind];
  return (
    <div className={cn("flex items-center gap-2", align === "right" && "flex-row-reverse")}>
      <span className={cn("flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-current/10", style.text)}>
        <Icon className="h-4 w-4" />
      </span>
      <h3 className="font-semibold text-heading">{name}</h3>
    </div>
  );
}
