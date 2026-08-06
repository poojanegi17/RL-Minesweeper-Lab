import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Hourglass } from "lucide-react";
import { Tabs } from "@/components/ui/Tabs";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { Skeleton } from "@/components/ui/Skeleton";
import { HyperparameterTable } from "@/components/experiment/HyperparameterTable";
import { AgentOverview } from "@/components/agents/AgentOverview";
import { AlgorithmPipeline } from "@/components/agents/AlgorithmPipeline";
import { ArchitectureDiagram } from "@/components/agents/ArchitectureDiagram";
import { DecisionExample } from "@/components/agents/DecisionExample";
import { AgentMetricsPanel } from "@/components/agents/AgentMetricsPanel";
import { getAgents } from "@/api/agents";
import { getExperiment } from "@/api/experiments";
import { getExperimentMetrics, getLeaderboard } from "@/api/metrics";
import { useApiQuery } from "@/hooks/useApiQuery";
import { agentKindFromName, slugifyAgentName } from "@/lib/agentAdapters";
import { isFamilySummary } from "@/lib/experimentAdapters";
import { AGENT_EXPLAINERS, CONCEPT_GLOSSARY } from "@/lib/agentExplainers";
import { AGENT_STYLES, type AgentStatus } from "@/data/types";
import type { ExperimentDetail, SummaryValue } from "@/types/experiment";
import type { LeaderboardEntry, MetricsResponse } from "@/types/metrics";
import type { Agent as ApiAgent } from "@/types/agent";

interface AgentDetailData {
  catalogAgent: ApiAgent;
  leaderboardEntry: LeaderboardEntry | undefined;
  /** null when this agent has no experiment artifacts, or the leaderboard
   * simply doesn't have an entry for it -- both render the same "no
   * training runs" empty states below. */
  experiment: ExperimentDetail | null;
  metrics: MetricsResponse | null;
}

/** Returns `null` (not an error) when the slug doesn't match any known agent
 * -- that's a normal 404-shaped outcome, not an API failure. */
async function fetchAgentDetail(slug: string): Promise<AgentDetailData | null> {
  const [agents, leaderboard] = await Promise.all([getAgents(), getLeaderboard()]);
  const catalogAgent = agents.find((agent) => slugifyAgentName(agent.name) === slug);
  if (!catalogAgent) return null;

  const leaderboardEntry = leaderboard.find((entry) => entry.agent === catalogAgent.name);

  let experiment: ExperimentDetail | null = null;
  let metrics: MetricsResponse | null = null;
  if (leaderboardEntry?.experiment_id) {
    const [detail, metricsResult] = await Promise.all([
      getExperiment(leaderboardEntry.experiment_id),
      getExperimentMetrics(leaderboardEntry.experiment_id),
    ]);
    // Leaderboard `experiment_id`s are always individual run ids, so this
    // resolves to `ExperimentDetail` -- guard anyway rather than assume.
    experiment = isFamilySummary(detail) ? null : detail;
    metrics = metricsResult;
  }

  return { catalogAgent, leaderboardEntry, experiment, metrics };
}

export function AgentDetail() {
  const { agentId } = useParams<{ agentId: string }>();
  const { data, status, error, retry } = useApiQuery(() => fetchAgentDetail(agentId ?? ""), [agentId]);

  if (status === "loading") {
    return (
      <div className="flex flex-col gap-8">
        <Skeleton className="h-5 w-32" />
        <Skeleton className="h-40 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (status === "error" && error) {
    return <ApiErrorState error={error} onRetry={retry} title="Couldn't load this agent" />;
  }

  if (status === "success" && data === null) {
    return (
      <EmptyState
        icon={Hourglass}
        title="Agent not found"
        description="This agent doesn't exist. Check the agent catalog for the current lineup."
        action={
          <Link to="/agents" className="text-sm text-primary hover:underline">
            Back to agents
          </Link>
        }
      />
    );
  }

  if (!data) return null; // unreachable given the branches above; narrows the type below

  const { catalogAgent, leaderboardEntry, experiment, metrics } = data;
  const agentStatus: AgentStatus = catalogAgent.has_experiment_artifacts ? "trained" : "baseline";
  const kind = agentKindFromName(catalogAgent.name);
  const style = AGENT_STYLES[kind];
  const explainer = AGENT_EXPLAINERS[kind];
  const concepts = explainer.concepts.map((label) => ({ label, description: CONCEPT_GLOSSARY[label] ?? "" }));

  return (
    <div className="flex flex-col gap-8">
      <Link
        to="/agents"
        className="flex w-fit items-center gap-1.5 text-sm text-text-muted transition-colors hover:text-text"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to agents
      </Link>

      <AgentOverview
        name={catalogAgent.name}
        kind={kind}
        status={agentStatus}
        description={catalogAgent.description}
        tagline={explainer.tagline}
        concepts={concepts}
        recordedTechniques={experiment?.techniques ?? []}
        leaderboardEntry={leaderboardEntry}
      />

      <section>
        <h2 className="mb-4 text-sm font-medium tracking-wide text-text-muted uppercase">How it decides</h2>
        {explainer.architecture ? (
          <ArchitectureDiagram architecture={explainer.architecture} accentClassName={style.text} />
        ) : explainer.pipeline ? (
          <AlgorithmPipeline steps={explainer.pipeline} loops={explainer.pipelineLoops} accentClassName={style.text} />
        ) : null}
      </section>

      <section>
        <h2 className="mb-4 text-sm font-medium tracking-wide text-text-muted uppercase">See it in action</h2>
        <DecisionExample config={explainer.decisionExample} />
      </section>

      <section>
        <h2 className="mb-4 text-sm font-medium tracking-wide text-text-muted uppercase">Training metrics</h2>
        <AgentMetricsPanel experiment={experiment} metrics={metrics} leaderboardEntry={leaderboardEntry} />
      </section>

      {experiment && (
        <section>
          <h2 className="mb-4 text-sm font-medium tracking-wide text-text-muted uppercase">Reference</h2>
          <Card>
            <Tabs tabs={[{ id: "hyperparameters", label: "Hyperparameters" }, { id: "training", label: "Training config" }]}>
              {(active) => (
                <>
                  {active === "hyperparameters" && <HyperparameterTable data={experiment.hyperparameters} />}
                  {active === "training" && (
                    <div className="flex flex-col gap-6">
                      <HyperparameterTable data={experiment.training_configuration} />
                      {experiment.best_checkpoint && (
                        <div>
                          <h3 className="mb-2 text-xs font-medium tracking-wide text-text-muted uppercase">
                            Best checkpoint
                          </h3>
                          <HyperparameterTable data={experiment.best_checkpoint as unknown as Record<string, SummaryValue>} />
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}
            </Tabs>
          </Card>
        </section>
      )}
    </div>
  );
}
