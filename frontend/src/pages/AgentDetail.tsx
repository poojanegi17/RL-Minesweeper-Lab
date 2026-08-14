import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useTheme } from "@/app/ThemeProvider";
import { ArrowLeft, Hourglass, Info } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { AgentOverview } from "@/components/agents/AgentOverview";
import { AlgorithmPipeline } from "@/components/agents/AlgorithmPipeline";
import { ArchitectureDiagram } from "@/components/agents/ArchitectureDiagram";
import { AgentConfigShowcase } from "@/components/agent/AgentConfigShowcase";
import { BestModelCard } from "@/components/agent/BestModelCard";
import { getAgents } from "@/api/agents";
import { getExperiment } from "@/api/experiments";
import { getLeaderboard } from "@/api/metrics";
import { useApiQuery } from "@/hooks/useApiQuery";
import { agentKindFromName, slugifyAgentName } from "@/lib/agentAdapters";
import { isFamilySummary } from "@/lib/experimentAdapters";
import { AGENT_EXPLAINERS, CONCEPT_GLOSSARY, NO_TRAINED_MODEL_EXPLANATION } from "@/lib/agentExplainers";
import { type FirstClickPolicy } from "@/lib/boardLevelQuery";
import { LEVEL_PIPELINE_IDS, type PipelineLevel } from "@/lib/levelPipelines";
import { AGENT_HEX, AGENT_STYLES, type AgentStatus } from "@/data/types";
import type { ExperimentDetail } from "@/types/experiment";
import type { LeaderboardEntry } from "@/types/metrics";
import type { Agent as ApiAgent } from "@/types/agent";

interface AgentDetailData {
  catalogAgent: ApiAgent;
  leaderboardEntry: LeaderboardEntry | undefined;
  /** null when this agent has no experiment artifacts, or the leaderboard
   * simply doesn't have an entry for it -- `recordedTechniques` on
   * `AgentOverview` just renders empty in either case. */
  experiment: ExperimentDetail | null;
}

/** Returns `null` (not an error) when the slug doesn't match any known agent
 * -- that's a normal 404-shaped outcome, not an API failure. */
async function fetchAgentDetail(slug: string): Promise<AgentDetailData | null> {
  const [agents, leaderboard] = await Promise.all([getAgents(), getLeaderboard()]);
  const catalogAgent = agents.find((agent) => slugifyAgentName(agent.name) === slug);
  if (!catalogAgent) return null;

  const leaderboardEntry = leaderboard.find((entry) => entry.agent === catalogAgent.name);

  let experiment: ExperimentDetail | null = null;
  if (leaderboardEntry?.experiment_id) {
    const detail = await getExperiment(leaderboardEntry.experiment_id);
    // Leaderboard `experiment_id`s are always individual run ids, so this
    // resolves to `ExperimentDetail` -- guard anyway rather than assume.
    experiment = isFamilySummary(detail) ? null : detail;
  }

  return { catalogAgent, leaderboardEntry, experiment };
}

export function AgentDetail() {
  const { agentId } = useParams<{ agentId: string }>();
  const { data, status, error, isSlow, retry } = useApiQuery(() => fetchAgentDetail(agentId ?? ""), [agentId]);
  // Called before the early returns below -- hooks must run in the same order
  // on every render, and the loading/error branches return before this point.
  const { theme } = useTheme();
  // Owned here, not inside `AgentConfigShowcase`, so this stays a single
  // selection driving that section rather than two independently-toggled
  // copies of the same state.
  const [policy, setPolicy] = useState<FirstClickPolicy>("area");
  const [level, setLevel] = useState("beginner");
  const [density, setDensity] = useState("standard");

  if (status === "loading") {
    return (
      <div className="flex flex-col gap-8">
        <Skeleton className="h-5 w-32" />
        <Skeleton className="h-40 w-full" />
        <Skeleton className="h-64 w-full" />
        {isSlow && <ColdStartNotice />}
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

  const { catalogAgent, experiment } = data;
  const agentStatus: AgentStatus = catalogAgent.has_experiment_artifacts ? "trained" : "baseline";
  const kind = agentKindFromName(catalogAgent.name);
  const style = AGENT_STYLES[kind];
  const explainer = AGENT_EXPLAINERS[kind];
  const concepts = explainer.concepts.map((label) => ({ label, description: CONCEPT_GLOSSARY[label] ?? "" }));
  // The real experiment/family id backing this agent's variant story at the
  // *currently selected* level -- undefined for a level nobody's trained at
  // yet (e.g. DQN has no Expert pipeline), matching `LevelDetail`'s own guard.
  const levelId = LEVEL_PIPELINE_IDS[catalogAgent.name]?.[level as PipelineLevel];

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
        <h2 className="text-sm font-medium tracking-wide text-text-muted uppercase">At a given board</h2>
        <p className="mt-1 mb-4 max-w-2xl text-sm text-text-muted">
          Pick an environment, board size and mine density to see how {catalogAgent.name} does there, and watch a
          recorded episode of it playing that exact configuration.
        </p>
        <AgentConfigShowcase
          agentName={catalogAgent.name}
          accentColor={AGENT_HEX[kind][theme]}
          policy={policy}
          level={level}
          density={density}
          onPolicyChange={setPolicy}
          onLevelChange={setLevel}
          onDensityChange={setDensity}
        />

        {levelId && (
          <div className="mt-6">
            <h3 className="mb-4 text-xs font-medium tracking-wide text-text-muted uppercase">
              Best found model at this board size and opening rule
            </h3>
            {/* The exact same collapsible detail card `/research/{agent}/{level}`
             * shows for its best-performing run -- `VariantStoryCard`'s real
             * hyperparameters, training setup, training curves and per-density
             * results, not a reimplementation of it (see `BestModelCard`).
             * Deliberately not passed `density`: one model is trained per
             * (level, opening rule), at Standard, so there is nothing for the
             * density selector to switch between here. */}
            <BestModelCard
              agentName={catalogAgent.name}
              level={level as PipelineLevel}
              levelId={levelId}
              policy={policy}
              accentColor={AGENT_HEX[kind][theme]}
            />
          </div>
        )}
      </section>

      {!catalogAgent.has_experiment_artifacts && NO_TRAINED_MODEL_EXPLANATION[kind] && (
        <section>
          <h2 className="mb-4 text-sm font-medium tracking-wide text-text-muted uppercase">Best agent at this configuration</h2>
          <Card className="flex items-start gap-3">
            <Info className="mt-0.5 h-4 w-4 shrink-0 text-text-muted" />
            <p className="text-sm text-text-muted">
              No best agent at this configuration — {NO_TRAINED_MODEL_EXPLANATION[kind]}
            </p>
          </Card>
        </section>
      )}
    </div>
  );
}
