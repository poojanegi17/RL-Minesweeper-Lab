import { Card } from "@/components/ui/Card";
import { FIRST_CLICK_POLICY_BLURBS, FIRST_CLICK_POLICY_LABELS, type FirstClickPolicy } from "@/lib/boardLevelQuery";
import { cn } from "@/lib/cn";

interface EnvironmentToggleProps {
  value: FirstClickPolicy;
  onChange: (policy: FirstClickPolicy) => void;
}

const ORDER: FirstClickPolicy[] = ["none", "area"];

const SUBTITLES: Record<FirstClickPolicy, string> = {
  none: "Original environment",
  area: "New environment",
};

/**
 * Which board distribution everything below is measured under.
 *
 * Deliberately larger and higher than the board-size and density controls,
 * because it is a different kind of choice. Size and density vary the
 * configuration within one game; `first_click_safe` changes the game itself --
 * under "none" roughly a fifth of 5x5 episodes are decided by the opening
 * click before any agent has acted. Win rates either side of this control are
 * therefore not two readings of the same quantity, which the caption says
 * outright rather than leaving to be inferred.
 */
export function EnvironmentToggle({ value, onChange }: EnvironmentToggleProps) {
  return (
    <Card className="flex flex-col items-center gap-3 text-center">
      <p className="text-xs font-medium tracking-wide text-text-muted uppercase">Environment</p>

      <div className="inline-flex rounded-xl border border-border p-1" role="group" aria-label="Board distribution">
        {ORDER.map((policy) => {
          const active = value === policy;
          return (
            <button
              key={policy}
              type="button"
              onClick={() => onChange(policy)}
              aria-pressed={active}
              className={cn(
                "flex min-w-[9rem] flex-col items-center rounded-lg px-5 py-2.5 transition-colors",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60",
                active ? "bg-primary text-white shadow-sm" : "text-text-muted hover:text-text",
              )}
            >
              <span className="text-sm font-semibold">{FIRST_CLICK_POLICY_LABELS[policy]}</span>
              <span className={cn("mt-0.5 text-[11px]", active ? "text-white/80" : "text-text-muted")}>
                {SUBTITLES[policy]}
              </span>
            </button>
          );
        })}
      </div>

      <p className="max-w-2xl text-sm text-text-muted">{FIRST_CLICK_POLICY_BLURBS[value]}</p>
      <p className="max-w-2xl text-xs text-text-muted">
        These are two different games, not two difficulty settings — a win rate under one cannot be read against a win
        rate under the other.
      </p>
    </Card>
  );
}
