import { LineChart } from "lucide-react";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { ExperimentMetricsChart } from "@/components/charts/ExperimentMetricsChart";
import { getExperimentMetrics } from "@/api/metrics";
import { useApiQuery } from "@/hooks/useApiQuery";

interface VariantTrainingCurvesProps {
  /** Real experiment id (`RunBrief.id`) -- this variant's own per-episode
   * training history, not a shared/representative one. */
  runId: string;
}

/**
 * This variant's real training curves (loss, TD-error, Q-values, rolling win
 * rate -- whichever series it actually logged), charted with the same
 * `ExperimentMetricsChart` grid `AgentDetail`/`ExperimentDetail` already use
 * elsewhere (each chart captioned with what it depicts, via
 * `metricSeriesConfig.ts`'s `description`), reused here per variant instead
 * of duplicated. Makes concrete what the card's prose already claims (e.g.
 * baseline's loss "spikes unpredictably late in training," LR decay's stays
 * "tightly bounded") -- seeing the curve is a different kind of evidence than
 * reading the sentence.
 *
 * Fetched when this mounts, i.e. only once its variant card is actually
 * opened (`VariantFlowCard` only renders `VariantStoryCard`, and this inside
 * it, when `isOpen`) -- not eagerly for all variants in the family at once.
 */
export function VariantTrainingCurves({ runId }: VariantTrainingCurvesProps) {
  const { data: metrics, status, error, isSlow, retry } = useApiQuery(() => getExperimentMetrics(runId), [runId]);

  return (
    <div>
      <p className="mb-2 flex items-center gap-2 text-xs font-medium tracking-wide text-text-muted uppercase">
        <LineChart className="h-4 w-4 shrink-0" aria-hidden="true" />
        Training curves
      </p>

      {status === "loading" && (
        <div className="flex flex-col gap-2">
          <Skeleton className="h-40 w-full" />
          {isSlow && <ColdStartNotice />}
        </div>
      )}
      {status === "error" && error && <ApiErrorState error={error} onRetry={retry} title="Couldn't load training curves" />}
      {status === "success" && metrics && <ExperimentMetricsChart metrics={metrics} />}
    </div>
  );
}
