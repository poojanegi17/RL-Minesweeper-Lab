import { Trophy } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { AgentMetricsPanel } from "@/components/agents/AgentMetricsPanel";
import { BoardConfigComparisonTable } from "@/components/research/BoardConfigComparisonTable";
import { HyperparameterTable } from "@/components/experiment/HyperparameterTable";
import { InvestigationNarrative } from "@/components/experiment/InvestigationNarrative";
import { VariantComparisonTable } from "@/components/experiment/VariantComparisonTable";
import {
  bestRun,
  buildInvestigationNarrative,
  buildVariantStories,
  deriveWhyItWon,
  formatPercent,
  formatReward,
  humanizeVariant,
} from "@/lib/experimentAdapters";
import type { AgentKind } from "@/data/types";
import type { ExperimentDetail, ExperimentSummary, SummaryValue } from "@/types/experiment";
import type { LeaderboardEntry, MetricsResponse } from "@/types/metrics";

interface ExperimentComparisonProps {
  agentName: string;
  kind: AgentKind;
  family: ExperimentSummary | null;
  variantDetails: ExperimentDetail[];
  detail: ExperimentDetail | null;
  metrics: MetricsResponse | null;
  leaderboardEntry: LeaderboardEntry | undefined;
  accentColor: string;
}

/**
 * "What did we measure?" -- the win-rate chart, the full variant table, and
 * the training-history chart the user asked to keep, framed as a research
 * conclusion rather than a bare dashboard: a synthesis sentence
 * (`InvestigationNarrative`) and a best-configuration highlight lead the
 * section, both derived from real `ExperimentSummary`/`RunBrief` fields.
 * `BoardConfigComparisonTable` (every board size/density this agent has been
 * evaluated at) always renders at the end, family or not -- Random/CSP have
 * no experiment family at all, but do have real board-size results.
 */
export function ExperimentComparison({ agentName, kind, family, variantDetails, detail, metrics, leaderboardEntry, accentColor }: ExperimentComparisonProps) {
  if (!family) {
    return (
      <div className="flex flex-col gap-6">
        <AgentMetricsPanel experiment={detail} metrics={metrics} leaderboardEntry={leaderboardEntry} />
        <BoardConfigComparisonTable agentName={agentName} kind={kind} accentColor={accentColor} />
      </div>
    );
  }

  const narrative = buildInvestigationNarrative(family);
  const best = bestRun(family);
  const stories = buildVariantStories(family.runs, variantDetails);
  const bestStory = best ? stories.find((story) => story.runBrief.id === best.id) : undefined;
  const baselineRun = family.runs.find((run) => run.variant === "baseline") ?? family.runs[0];
  const whyItWon = bestStory ? deriveWhyItWon(bestStory, narrative) : "";
  const showBestCheckpoint = best && detail?.id === best.id && detail?.best_checkpoint;

  return (
    <div className="flex flex-col gap-6">
      {narrative && <InvestigationNarrative narrative={narrative} />}

      {best && (
        <Card
          className="border-white/10 bg-gradient-to-b from-primary/10 to-primary/[0.03] shadow-lg shadow-black/[0.06] backdrop-blur-sm"
          style={{ borderTop: `3px solid ${accentColor}` }}
        >
          <div className="flex items-center gap-2" style={{ color: accentColor }}>
            <Trophy className="h-4 w-4" />
            <h4 className="text-xs font-medium tracking-wide uppercase">Best configuration</h4>
          </div>
          <p className="mt-2 text-lg font-semibold text-heading">
            {agentName} — {best.variant ? humanizeVariant(best.variant) : best.title}
          </p>
          <dl className="mt-3 flex flex-wrap gap-6 text-sm">
            <div>
              <dt className="text-xs text-text-muted">Win rate</dt>
              <dd className="font-mono font-medium text-text">
                {baselineRun && baselineRun.id !== best.id ? `${formatPercent(baselineRun.win_rate)} → ` : ""}
                {formatPercent(best.win_rate)}
              </dd>
            </div>
            {bestStory?.winRateImprovementPct != null && (
              <div>
                <dt className="text-xs text-text-muted">Improvement</dt>
                <dd className="font-mono font-medium text-emerald-500">
                  {bestStory.winRateImprovementPct >= 0 ? "+" : ""}
                  {bestStory.winRateImprovementPct.toFixed(0)}%
                </dd>
              </div>
            )}
            <div>
              <dt className="text-xs text-text-muted">Avg. reward</dt>
              <dd className="font-mono font-medium text-text">{formatReward(best.avg_reward)}</dd>
            </div>
            <div>
              <dt className="text-xs text-text-muted">Training budget</dt>
              <dd className="font-mono font-medium text-text">
                {best.episodes.toLocaleString()} episodes
                {baselineRun && baselineRun.id !== best.id
                  ? baselineRun.episodes === best.episodes
                    ? " (same as baseline)"
                    : ` (vs. baseline's ${baselineRun.episodes.toLocaleString()})`
                  : ""}
              </dd>
            </div>
          </dl>
          {whyItWon && (
            <div className="mt-3">
              <p className="text-xs font-medium tracking-wide text-text-muted uppercase">Why it won</p>
              <p className="mt-1 text-sm text-text">{whyItWon}</p>
            </div>
          )}
          {showBestCheckpoint && (
            <div className="mt-3">
              <p className="mb-1.5 text-xs font-medium tracking-wide text-text-muted uppercase">Best checkpoint</p>
              <HyperparameterTable data={detail!.best_checkpoint as unknown as Record<string, SummaryValue>} />
            </div>
          )}
        </Card>
      )}

      <VariantComparisonTable runs={family.runs} currentRunId={detail?.id} agent={agentName} />

      <AgentMetricsPanel experiment={detail} metrics={metrics} leaderboardEntry={leaderboardEntry} />

      <BoardConfigComparisonTable agentName={agentName} kind={kind} accentColor={accentColor} />
    </div>
  );
}
