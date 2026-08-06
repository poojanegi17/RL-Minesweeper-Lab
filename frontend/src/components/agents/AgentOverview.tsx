import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { AgentStatusBadge } from "@/components/agent/AgentStatusBadge";
import { AGENT_ICONS } from "@/components/agent/agentIcons";
import { CONCEPT_GLOSSARY } from "@/lib/agentExplainers";
import { AGENT_STYLES, type AgentKind, type AgentStatus } from "@/data/types";
import type { LeaderboardEntry } from "@/types/metrics";
import { cn } from "@/lib/cn";

interface AgentOverviewProps {
  name: string;
  kind: AgentKind;
  status: AgentStatus;
  /** Real catalog description (`GET /api/agents`). */
  description: string;
  /** One-line "how it decides" summary, from `@/lib/agentExplainers`. */
  tagline: string;
  /** Algorithm-level facts this agent always exhibits (resolved from `CONCEPT_GLOSSARY`). */
  concepts: { label: string; description: string }[];
  /** Real techniques the best recorded run actually used (`ExperimentDetail.techniques`),
   * empty for agents with no experiment artifacts (CSP, Q-Learning, Random). */
  recordedTechniques: string[];
  leaderboardEntry: LeaderboardEntry | undefined;
}

/**
 * The page's hero/summary card: identity, a punchy one-line explanation, the
 * headline win-rate stat (tagged live vs. reference), and two distinct
 * technique lists that are careful not to blur together -- "How it works"
 * (always-true algorithm facts) vs. "Recorded in this run" (only what the
 * best actual run's data confirms).
 */
export function AgentOverview({
  name,
  kind,
  status,
  description,
  tagline,
  concepts,
  recordedTechniques,
  leaderboardEntry,
}: AgentOverviewProps) {
  const style = AGENT_STYLES[kind];
  const Icon = AGENT_ICONS[kind];
  const winRate = leaderboardEntry?.win_rate ?? null;

  return (
    <Card className="flex flex-col gap-6">
      <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-4">
          <span className={cn("flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-current/10", style.text)}>
            <Icon className="h-6 w-6" />
          </span>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-3xl font-semibold tracking-tight text-heading">{name}</h1>
              <AgentStatusBadge status={status} />
            </div>
            <p className="mt-1.5 max-w-xl font-medium text-text">{tagline}</p>
            <p className="mt-1 max-w-xl text-sm text-text-muted">{description}</p>
          </div>
        </div>

        {winRate != null && (
          <div className="shrink-0 rounded-xl border border-border bg-surface-hover/60 px-4 py-3 text-center">
            <p className="font-mono text-2xl font-semibold text-heading">{(winRate * 100).toFixed(1)}%</p>
            <p className="mt-0.5 text-xs text-text-muted">
              Win rate
              <Badge variant="outline" className="ml-1.5 align-middle">
                {leaderboardEntry?.source === "experiment_artifact" ? "Live" : "Reference"}
              </Badge>
            </p>
          </div>
        )}
      </div>

      {concepts.length > 0 && (
        <div className="border-t border-border pt-5">
          <h3 className="mb-3 text-xs font-medium tracking-wide text-text-muted uppercase">How it works</h3>
          <dl className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
            {concepts.map((concept) => (
              <div key={concept.label}>
                <dt className="text-sm font-medium text-heading">{concept.label}</dt>
                <dd className="mt-0.5 text-sm text-text-muted">{concept.description}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      {recordedTechniques.length > 0 && (
        <div className="border-t border-border pt-5">
          <h3 className="mb-2 text-xs font-medium tracking-wide text-text-muted uppercase">Recorded in this run</h3>
          <ul className="flex flex-wrap gap-1.5">
            {recordedTechniques.map((technique) => (
              <li key={technique}>
                <Badge variant="outline" title={CONCEPT_GLOSSARY[technique]}>
                  {technique}
                </Badge>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}
