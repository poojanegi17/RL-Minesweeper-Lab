import { Hourglass } from "lucide-react";
import { EmptyState } from "@/components/ui/EmptyState";
import type { SummaryValue } from "@/types/experiment";

interface HyperparameterTableProps {
  data: Record<string, SummaryValue> | undefined;
  emptyTitle?: string;
  emptyDescription?: string;
}

/**
 * A generic key/value table for whatever a summary JSON happens to record --
 * used for `hyperparameters`, `training_configuration`, and `best_checkpoint`
 * alike (all three are `Record<string, SummaryValue>`-shaped), so it's
 * reused across those sections rather than duplicated per section.
 *
 * Shared by `AgentDetail` and `ExperimentDetail`.
 */
export function HyperparameterTable({
  data,
  emptyTitle = "Nothing recorded",
  emptyDescription = "No values were found for this section.",
}: HyperparameterTableProps) {
  if (!data || Object.keys(data).length === 0) {
    return <EmptyState icon={Hourglass} title={emptyTitle} description={emptyDescription} />;
  }

  return (
    <dl className="flex max-w-xl flex-col gap-2">
      {Object.entries(data).map(([key, value]) => (
        <div key={key} className="flex items-center justify-between gap-4 text-sm">
          <dt className="text-text-muted">{key.replace(/_/g, " ")}</dt>
          <dd className="font-mono font-medium text-text">{formatValue(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Exported for reuse anywhere else a raw `SummaryValue` needs the same
 * number/boolean/array/null formatting this table already applies per cell
 * (e.g. `VariantStoryCard`'s "changed from baseline" diff). */
export function formatValue(value: SummaryValue): string {
  if (value === null) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return value.toLocaleString(undefined, { maximumFractionDigits: 6 });
  if (Array.isArray(value)) return JSON.stringify(value);
  return String(value);
}
