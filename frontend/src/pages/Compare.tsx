import { useEffect, useState } from "react";
import { ArrowLeftRight } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { Button } from "@/components/ui/Button";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { HyperparameterTable } from "@/components/experiment/HyperparameterTable";
import { ExperimentMetricsChart } from "@/components/charts/ExperimentMetricsChart";
import { AgentComparisonChart } from "@/components/compare/AgentComparisonChart";
import { CompareStatRow } from "@/components/compare/CompareStatRow";
import { getAgents } from "@/api/agents";
import { getExperiment } from "@/api/experiments";
import { getExperimentMetrics, getLeaderboard } from "@/api/metrics";
import { useApiQuery } from "@/hooks/useApiQuery";
import { useTheme } from "@/app/ThemeProvider";
import { agentKindFromName } from "@/lib/agentAdapters";
import { isFamilySummary } from "@/lib/experimentAdapters";
import { AGENT_HEX } from "@/data/types";
import type { ExperimentDetail } from "@/types/experiment";
import type { LeaderboardEntry, MetricsResponse } from "@/types/metrics";
import type { Agent as ApiAgent } from "@/types/agent";

interface Catalog {
  agents: ApiAgent[];
  leaderboard: LeaderboardEntry[];
}

async function fetchCatalog(): Promise<Catalog> {
  const [agents, leaderboard] = await Promise.all([getAgents(), getLeaderboard()]);
  return { agents, leaderboard };
}

interface AgentCompareData {
  catalogAgent: ApiAgent;
  leaderboardEntry: LeaderboardEntry | undefined;
  experiment: ExperimentDetail | null;
  metrics: MetricsResponse | null;
}

/** Resolves one agent's full comparison data -- same leaderboard-entry ->
 * experiment resolution `AgentDetail.tsx`'s `fetchAgentDetail` uses, just
 * fetched twice (once per side) instead of once. */
async function fetchAgentCompareData(name: string, catalog: Catalog): Promise<AgentCompareData | null> {
  const catalogAgent = catalog.agents.find((agent) => agent.name === name);
  if (!catalogAgent) return null;

  const leaderboardEntry = catalog.leaderboard.find((entry) => entry.agent === name);

  let experiment: ExperimentDetail | null = null;
  let metrics: MetricsResponse | null = null;
  if (leaderboardEntry?.experiment_id) {
    const [detail, metricsResult] = await Promise.all([
      getExperiment(leaderboardEntry.experiment_id),
      getExperimentMetrics(leaderboardEntry.experiment_id),
    ]);
    experiment = isFamilySummary(detail) ? null : detail;
    metrics = metricsResult;
  }

  return { catalogAgent, leaderboardEntry, experiment, metrics };
}

async function fetchBothAgents(nameA: string, nameB: string, catalog: Catalog) {
  const [a, b] = await Promise.all([fetchAgentCompareData(nameA, catalog), fetchAgentCompareData(nameB, catalog)]);
  return { a, b };
}

/**
 * Put any two agents side by side: win rate / avg reward / avg episode
 * length, an overlaid learning-curve chart (only agents with a per-episode
 * history contribute a line -- Random/CSP/Q-Learning honestly show "no
 * training history" instead of a fabricated flat line), each agent's full
 * metric grid, and hyperparameters/training config. Every number here comes
 * from the same endpoints `AgentDetail`/`Research` already use -- no new
 * backend surface for this page.
 */
export function Compare() {
  const { theme } = useTheme();
  const { data: catalog, status: catalogStatus, error: catalogError, isSlow: catalogSlow, retry: retryCatalog } = useApiQuery(fetchCatalog, []);

  const [nameA, setNameA] = useState<string | null>(null);
  const [nameB, setNameB] = useState<string | null>(null);

  // Default to the leaderboard's top two win-rate entries once it loads --
  // derived from live data, never hardcoded, so it stays correct as results change.
  useEffect(() => {
    if (!catalog || (nameA !== null && nameB !== null)) return;
    const ranked = [...catalog.leaderboard]
      .filter((entry) => entry.win_rate != null)
      .sort((x, y) => (y.win_rate ?? 0) - (x.win_rate ?? 0));
    setNameA((current) => current ?? ranked[0]?.agent ?? catalog.agents[0]?.name ?? null);
    setNameB((current) => current ?? ranked[1]?.agent ?? catalog.agents[1]?.name ?? null);
  }, [catalog, nameA, nameB]);

  const {
    data: pair,
    status: pairStatus,
    error: pairError,
    isSlow: pairSlow,
    retry: retryPair,
  } = useApiQuery(() => (nameA && nameB && catalog ? fetchBothAgents(nameA, nameB, catalog) : Promise.resolve(null)), [nameA, nameB, catalog]);

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight text-heading">Compare agents</h1>
        <p className="mt-2 max-w-2xl text-text-muted">
          Pick any two agents to see them side by side -- win rate, learning curves, and every other metric each one
          actually recorded.
        </p>
      </div>

      {catalogStatus === "loading" && (
        <div className="flex flex-col gap-4">
          <Skeleton className="h-12 w-full max-w-xl" />
          <Skeleton className="h-40 w-full" />
          {catalogSlow && <ColdStartNotice />}
        </div>
      )}

      {catalogStatus === "error" && catalogError && (
        <ApiErrorState error={catalogError} onRetry={retryCatalog} title="Couldn't load the agent catalog" />
      )}

      {catalogStatus === "success" && catalog && nameA && nameB && (
        <>
          <AgentPicker
            agents={catalog.agents}
            nameA={nameA}
            nameB={nameB}
            onChangeA={setNameA}
            onChangeB={setNameB}
            onSwap={() => {
              setNameA(nameB);
              setNameB(nameA);
            }}
          />

          {pairStatus === "loading" && (
            <div className="flex flex-col gap-4">
              <Skeleton className="h-32 w-full" />
              <Skeleton className="h-72 w-full" />
              {pairSlow && <ColdStartNotice />}
            </div>
          )}

          {pairStatus === "error" && pairError && (
            <ApiErrorState error={pairError} onRetry={retryPair} title="Couldn't load these agents" />
          )}

          {pairStatus === "success" && pair?.a && pair?.b && (
            <CompareResults agentA={pair.a} agentB={pair.b} theme={theme} />
          )}
        </>
      )}
    </div>
  );
}

function AgentPicker({
  agents,
  nameA,
  nameB,
  onChangeA,
  onChangeB,
  onSwap,
}: {
  agents: ApiAgent[];
  nameA: string;
  nameB: string;
  onChangeA: (name: string) => void;
  onChangeB: (name: string) => void;
  onSwap: () => void;
}) {
  return (
    <div className="flex flex-col items-center gap-3 sm:flex-row">
      <Select className="sm:max-w-xs" aria-label="Agent A" value={nameA} onChange={(e) => onChangeA(e.target.value)}>
        {agents
          .filter((agent) => agent.name !== nameB)
          .map((agent) => (
            <option key={agent.name} value={agent.name}>
              {agent.name}
            </option>
          ))}
      </Select>

      <Button variant="secondary" size="sm" onClick={onSwap} aria-label="Swap agents">
        <ArrowLeftRight className="h-3.5 w-3.5" />
      </Button>

      <Select className="sm:max-w-xs" aria-label="Agent B" value={nameB} onChange={(e) => onChangeB(e.target.value)}>
        {agents
          .filter((agent) => agent.name !== nameA)
          .map((agent) => (
            <option key={agent.name} value={agent.name}>
              {agent.name}
            </option>
          ))}
      </Select>
    </div>
  );
}

function CompareResults({ agentA, agentB, theme }: { agentA: AgentCompareData; agentB: AgentCompareData; theme: "light" | "dark" }) {
  const kindA = agentKindFromName(agentA.catalogAgent.name);
  const kindB = agentKindFromName(agentB.catalogAgent.name);

  return (
    <div className="flex flex-col gap-6">
      <CompareStatRow
        agentA={{ name: agentA.catalogAgent.name, kind: kindA, leaderboardEntry: agentA.leaderboardEntry }}
        agentB={{ name: agentB.catalogAgent.name, kind: kindB, leaderboardEntry: agentB.leaderboardEntry }}
      />

      <section>
        <h2 className="mb-4 text-sm font-medium tracking-wide text-text-muted uppercase">Learning curves</h2>
        <Card>
          <AgentComparisonChart
            agentA={{ name: agentA.catalogAgent.name, kind: kindA, metrics: agentA.metrics }}
            agentB={{ name: agentB.catalogAgent.name, kind: kindB, metrics: agentB.metrics }}
            colorA={AGENT_HEX[kindA][theme]}
            colorB={AGENT_HEX[kindB][theme]}
          />
        </Card>
      </section>

      <section>
        <h2 className="mb-4 text-sm font-medium tracking-wide text-text-muted uppercase">Every recorded metric</h2>
        <div className="grid gap-4 lg:grid-cols-2">
          <AgentMetricsColumn name={agentA.catalogAgent.name} metrics={agentA.metrics} />
          <AgentMetricsColumn name={agentB.catalogAgent.name} metrics={agentB.metrics} />
        </div>
      </section>

      {(agentA.experiment || agentB.experiment) && (
        <section>
          <h2 className="mb-4 text-sm font-medium tracking-wide text-text-muted uppercase">Hyperparameters</h2>
          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <h3 className="mb-3 text-xs font-medium tracking-wide text-text-muted uppercase">{agentA.catalogAgent.name}</h3>
              <HyperparameterTable
                data={agentA.experiment?.hyperparameters}
                emptyTitle="No hyperparameters recorded"
                emptyDescription={`${agentA.catalogAgent.name} has no experiment artifact to read hyperparameters from.`}
              />
            </Card>
            <Card>
              <h3 className="mb-3 text-xs font-medium tracking-wide text-text-muted uppercase">{agentB.catalogAgent.name}</h3>
              <HyperparameterTable
                data={agentB.experiment?.hyperparameters}
                emptyTitle="No hyperparameters recorded"
                emptyDescription={`${agentB.catalogAgent.name} has no experiment artifact to read hyperparameters from.`}
              />
            </Card>
          </div>
        </section>
      )}
    </div>
  );
}

function AgentMetricsColumn({ name, metrics }: { name: string; metrics: MetricsResponse | null }) {
  return (
    <Card>
      <h3 className="mb-3 text-xs font-medium tracking-wide text-text-muted uppercase">{name}</h3>
      {metrics ? (
        <ExperimentMetricsChart metrics={metrics} />
      ) : (
        <p className="text-sm text-text-muted">{name} has no per-episode training history recorded.</p>
      )}
    </Card>
  );
}
