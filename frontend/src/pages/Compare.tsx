import { useEffect, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { HyperparameterTable } from "@/components/experiment/HyperparameterTable";
import { ExperimentMetricsChart } from "@/components/charts/ExperimentMetricsChart";
import { AgentComparisonChart } from "@/components/compare/AgentComparisonChart";
import { EnvironmentCompareCard } from "@/components/compare/EnvironmentCompareCard";
import { EnvironmentToggle } from "@/components/compare/EnvironmentToggle";
import { getAgents } from "@/api/agents";
import { getExperimentMetrics, getLeaderboard } from "@/api/metrics";
import { useApiQuery } from "@/hooks/useApiQuery";
import { useTheme } from "@/app/ThemeProvider";
import { agentKindFromName } from "@/lib/agentAdapters";
import { humanizeVariant, type VariantStory } from "@/lib/experimentAdapters";
import {
  bestRunForEnv,
  fetchLevelPipeline,
  LEVEL_LABELS_FULL,
  LEVEL_PIPELINE_IDS,
  pipelineIdForEnv,
  POLICY_ENV_VERSION,
  visibleStories,
  type PipelineLevel,
} from "@/lib/levelPipelines";
import { FIRST_CLICK_POLICY_LABELS, type FirstClickPolicy } from "@/lib/boardLevelQuery";
import { AGENT_HEX } from "@/data/types";
import type { MetricsResponse } from "@/types/metrics";
import type { LeaderboardEntry } from "@/types/metrics";
import type { Agent as ApiAgent } from "@/types/agent";

interface Catalog {
  agents: ApiAgent[];
  leaderboard: LeaderboardEntry[];
}

async function fetchCatalog(): Promise<Catalog> {
  const [agents, leaderboard] = await Promise.all([getAgents(), getLeaderboard()]);
  return { agents, leaderboard };
}

/** Why an agent contributes no training-run detail, so the blank state can say
 * which of the two very different reasons applies. */
type Absence =
  /** Random/CSP/Q-Learning: no training artifacts anywhere, by construction. */
  | "no-artifacts"
  /** A learned agent with no run at *this* board size yet -- DQN and PPO at
   * Expert. Deliberately blank rather than falling back to another level's
   * run, which would caption this cell with a model never trained for it. */
  | "not-trained-here";

interface AgentCompareData {
  catalogAgent: ApiAgent;
  story: VariantStory | null;
  metrics: MetricsResponse | null;
  /** True when the shown run trained under the *other* distribution. */
  isTransfer: boolean;
  absence: Absence | null;
}

/**
 * One agent's training-run detail for the selected board size and environment.
 *
 * Resolved through the research pipeline (`LEVEL_PIPELINE_IDS` +
 * `bestRunForEnv`), which is the same path `/research/{agent}/{level}` and the
 * agent page's "best found model" card use -- so this page reports the run the
 * pipeline reports, and inherits its correctness for free: hidden variants stay
 * hidden, retired runs never surface, and a v2 selection resolves to the family
 * that actually trained under v2.
 *
 * It previously read `experiment_id` off the *unscoped* default leaderboard, so
 * every figure below the fold described the v1 beginner/standard run no matter
 * what the selectors said. That path could not have been made to work either:
 * `_board_result_entries` in the backend sets no `experiment_id` at all, so
 * outside beginner/standard there was nothing to resolve.
 */
async function fetchAgentCompareData(
  name: string,
  catalog: Catalog,
  level: PipelineLevel,
  policy: FirstClickPolicy,
): Promise<AgentCompareData | null> {
  const catalogAgent = catalog.agents.find((agent) => agent.name === name);
  if (!catalogAgent) return null;

  const blank = (absence: Absence): AgentCompareData => ({
    catalogAgent,
    story: null,
    metrics: null,
    isTransfer: false,
    absence,
  });

  if (!catalogAgent.has_experiment_artifacts) return blank("no-artifacts");

  const levelId = LEVEL_PIPELINE_IDS[name]?.[level];
  if (!levelId) return blank("not-trained-here");

  const env = POLICY_ENV_VERSION[policy];
  const envPipeline = pipelineIdForEnv(name, level, env);
  const pipeline = await fetchLevelPipeline(envPipeline?.id ?? levelId);

  // Matches `BestModelCard`: a dedicated env family is its own list, so the
  // level's v1 variant filters must not be applied to it.
  const shown = envPipeline ? pipeline.stories : visibleStories(pipeline.stories, name, level);
  const picked = bestRunForEnv(shown, env);
  if (!picked) return blank("not-trained-here");

  const metrics = picked.story.runBrief.metrics_available
    ? await getExperimentMetrics(picked.story.runBrief.id)
    : null;

  return { catalogAgent, story: picked.story, metrics, isTransfer: picked.isTransfer, absence: null };
}

async function fetchBothAgents(
  nameA: string,
  nameB: string,
  catalog: Catalog,
  level: PipelineLevel,
  policy: FirstClickPolicy,
) {
  const [a, b] = await Promise.all([
    fetchAgentCompareData(nameA, catalog, level, policy),
    fetchAgentCompareData(nameB, catalog, level, policy),
  ]);
  return { a, b };
}

/**
 * Put any two agents side by side: win rate / avg reward / avg episode
 * length at the selected board cell, an overlaid learning-curve chart (only
 * agents with a per-episode history contribute a line -- Random/CSP/Q-Learning
 * honestly show "no training history" instead of a fabricated flat line), each
 * agent's full metric grid, and hyperparameters/training config. Every number
 * here comes from the same endpoints `AgentDetail`/`Research` already use --
 * no new backend surface for this page.
 *
 * `level`/`density`/`policy` are owned here rather than by the child card,
 * because the training-run half below depends on level and policy too.
 */
export function Compare() {
  const { theme } = useTheme();
  const { data: catalog, status: catalogStatus, error: catalogError, isSlow: catalogSlow, retry: retryCatalog } = useApiQuery(fetchCatalog, []);

  const [policy, setPolicy] = useState<FirstClickPolicy>("none");
  const [level, setLevel] = useState<string>("beginner");
  const [density, setDensity] = useState<string>("standard");
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

  const pipelineLevel = level as PipelineLevel;

  const {
    data: pair,
    status: pairStatus,
    error: pairError,
    isSlow: pairSlow,
    retry: retryPair,
  } = useApiQuery(
    () =>
      nameA && nameB && catalog
        ? fetchBothAgents(nameA, nameB, catalog, pipelineLevel, policy)
        : Promise.resolve(null),
    [nameA, nameB, catalog, pipelineLevel, policy],
  );

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight text-heading">Compare agents</h1>
        <p className="mt-2 max-w-2xl text-text-muted">
          Pick an environment, a board size and density, then any two agents. Every figure below is measured under the
          settings selected at the top of the page.
        </p>
      </div>

      {catalogStatus === "success" && catalog && nameA && nameB && (
        <section className="flex flex-col gap-5">
          <EnvironmentToggle value={policy} onChange={setPolicy} />
          <EnvironmentCompareCard
            policy={policy}
            level={level}
            density={density}
            onLevelChange={setLevel}
            onDensityChange={setDensity}
            agentNames={catalog.agents.map((agent) => agent.name)}
            nameA={nameA}
            nameB={nameB}
            onChangeA={setNameA}
            onChangeB={setNameB}
          />
        </section>
      )}

      <div className="border-t border-border pt-8">
        <h2 className="text-xl font-semibold tracking-tight text-heading">Training run details</h2>
        <p className="mt-2 max-w-2xl text-text-muted">
          The best model each agent has at{" "}
          <span className="text-text">{LEVEL_LABELS_FULL[pipelineLevel] ?? level}</span> under{" "}
          <span className="text-text">{FIRST_CLICK_POLICY_LABELS[policy].toLowerCase()}</span> — the same run the
          research pipeline reports for that combination. Training happens at Standard density only, so this half
          follows the board size and the opening rule but not the density selector. Agents with no recorded history say
          so instead of showing a fabricated line.
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
            <CompareResults agentA={pair.a} agentB={pair.b} level={pipelineLevel} policy={policy} theme={theme} />
          )}
        </>
      )}
    </div>
  );
}

/** One line naming the run a column describes, or why there isn't one. Without
 * it the curves and hyperparameters are unattributed -- two agents' panels look
 * equivalent whether one is a matched result, a transfer, or absent. */
function RunCaption({ data, level, policy }: { data: AgentCompareData; level: PipelineLevel; policy: FirstClickPolicy }) {
  if (data.absence === "no-artifacts") {
    return (
      <p className="text-xs text-text-muted">
        {data.catalogAgent.name} does no training, so it has no run to report.
      </p>
    );
  }
  if (data.absence === "not-trained-here" || !data.story) {
    return (
      <p className="text-xs text-text-muted">
        Not trained at {LEVEL_LABELS_FULL[level] ?? level} yet.
      </p>
    );
  }
  const { runBrief } = data.story;
  return (
    <p className="text-xs text-text-muted">
      <span className="font-mono text-text">{runBrief.id}</span>
      {runBrief.variant ? ` · ${humanizeVariant(runBrief.variant)}` : ""}
      {data.isTransfer
        ? ` · trained under the other opening rule, shown here as a generalization result`
        : ` · trained under ${FIRST_CLICK_POLICY_LABELS[policy].toLowerCase()}`}
    </p>
  );
}

function CompareResults({
  agentA,
  agentB,
  level,
  policy,
  theme,
}: {
  agentA: AgentCompareData;
  agentB: AgentCompareData;
  level: PipelineLevel;
  policy: FirstClickPolicy;
  theme: "light" | "dark";
}) {
  const kindA = agentKindFromName(agentA.catalogAgent.name);
  const kindB = agentKindFromName(agentB.catalogAgent.name);

  return (
    <div className="flex flex-col gap-6">
      <section>
        <h2 className="mb-4 text-sm font-medium tracking-wide text-text-muted uppercase">Learning curves</h2>
        <Card className="flex flex-col gap-3">
          <AgentComparisonChart
            agentA={{ name: agentA.catalogAgent.name, kind: kindA, metrics: agentA.metrics }}
            agentB={{ name: agentB.catalogAgent.name, kind: kindB, metrics: agentB.metrics }}
            colorA={AGENT_HEX[kindA][theme]}
            colorB={AGENT_HEX[kindB][theme]}
          />
          <div className="grid gap-2 border-t border-border pt-3 sm:grid-cols-2">
            <RunCaption data={agentA} level={level} policy={policy} />
            <RunCaption data={agentB} level={level} policy={policy} />
          </div>
        </Card>
      </section>

      <section>
        <h2 className="mb-4 text-sm font-medium tracking-wide text-text-muted uppercase">Every recorded metric</h2>
        <div className="grid gap-4 lg:grid-cols-2">
          <AgentMetricsColumn data={agentA} level={level} policy={policy} />
          <AgentMetricsColumn data={agentB} level={level} policy={policy} />
        </div>
      </section>

      {(agentA.story || agentB.story) && (
        <section>
          <h2 className="mb-4 text-sm font-medium tracking-wide text-text-muted uppercase">Hyperparameters</h2>
          <div className="grid gap-4 lg:grid-cols-2">
            <HyperparameterColumn data={agentA} level={level} policy={policy} />
            <HyperparameterColumn data={agentB} level={level} policy={policy} />
          </div>
        </section>
      )}
    </div>
  );
}

function HyperparameterColumn({
  data,
  level,
  policy,
}: {
  data: AgentCompareData;
  level: PipelineLevel;
  policy: FirstClickPolicy;
}) {
  return (
    <Card className="flex flex-col gap-3">
      <div>
        <h3 className="text-xs font-medium tracking-wide text-text-muted uppercase">{data.catalogAgent.name}</h3>
        <div className="mt-1">
          <RunCaption data={data} level={level} policy={policy} />
        </div>
      </div>
      <HyperparameterTable
        data={data.story?.detail.hyperparameters}
        emptyTitle="No hyperparameters recorded"
        emptyDescription={
          data.absence === "not-trained-here"
            ? `${data.catalogAgent.name} has no run at ${LEVEL_LABELS_FULL[level] ?? level} to read hyperparameters from.`
            : `${data.catalogAgent.name} has no experiment artifact to read hyperparameters from.`
        }
      />
    </Card>
  );
}

function AgentMetricsColumn({
  data,
  level,
  policy,
}: {
  data: AgentCompareData;
  level: PipelineLevel;
  policy: FirstClickPolicy;
}) {
  return (
    <Card className="flex flex-col gap-3">
      <div>
        <h3 className="text-xs font-medium tracking-wide text-text-muted uppercase">{data.catalogAgent.name}</h3>
        <div className="mt-1">
          <RunCaption data={data} level={level} policy={policy} />
        </div>
      </div>
      {data.metrics ? (
        <ExperimentMetricsChart metrics={data.metrics} />
      ) : (
        <p className="text-sm text-text-muted">
          {data.absence === "not-trained-here"
            ? `No per-episode history — ${data.catalogAgent.name} has not been trained at ${LEVEL_LABELS_FULL[level] ?? level}.`
            : `${data.catalogAgent.name} has no per-episode training history recorded.`}
        </p>
      )}
    </Card>
  );
}
