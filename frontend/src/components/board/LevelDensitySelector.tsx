import { motion } from "framer-motion";
import { Dropdown } from "@/components/ui/Dropdown";
import { cn } from "@/lib/cn";
import type { BoardLevelInfo } from "@/types/boardConfig";

export const LEVEL_LABELS: Record<string, string> = {
  beginner: "Beginner",
  intermediate: "Intermediate",
  expert: "Expert",
};

export const DENSITY_LABELS: Record<string, string> = {
  sparse: "Sparse",
  standard: "Standard",
  dense: "Dense",
};

interface LevelDensitySelectorProps {
  configs: BoardLevelInfo[];
  level: string;
  density: string;
  onLevelChange: (level: string) => void;
  onDensityChange: (density: string) => void;
  /** Compact mode drops the "Board size" label and shrinks spacing, for
   * surfaces (like a home-page card) where a full-size control would crowd
   * everything else. */
  compact?: boolean;
}

/**
 * A toggle for board size (Beginner/Intermediate/Expert) with a mine-density
 * dropdown beneath it (Sparse/Standard/Dense for whichever size is
 * selected) -- built from the real `GET /api/board-configs` catalog, not a
 * hardcoded copy, so it can never drift out of sync with what the backend
 * actually serves.
 */
export function LevelDensitySelector({ configs, level, density, onLevelChange, onDensityChange, compact = false }: LevelDensitySelectorProps) {
  const activeConfig = configs.find((c) => c.level === level);
  const densityOptions = activeConfig ? Object.keys(activeConfig.densities) : [];

  function handleLevelChange(nextLevel: string) {
    onLevelChange(nextLevel);
    const nextConfig = configs.find((c) => c.level === nextLevel);
    if (nextConfig && !(density in nextConfig.densities)) {
      onDensityChange(Object.keys(nextConfig.densities)[0]);
    }
  }

  return (
    <div className={cn("flex flex-col gap-2", compact ? "gap-1.5" : "gap-2")}>
      {!compact && <span className="text-xs font-medium tracking-wide text-text-muted uppercase">Board size</span>}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1 rounded-lg border border-border bg-surface p-1">
          {configs.map((config) => {
            const active = config.level === level;
            return (
              <button
                key={config.level}
                type="button"
                onClick={() => handleLevelChange(config.level)}
                aria-pressed={active}
                className="relative rounded-md px-3 py-1.5 text-sm font-medium transition-colors"
              >
                {active && (
                  <motion.span
                    layoutId={compact ? "level-toggle-compact" : "level-toggle"}
                    className="absolute inset-0 rounded-md bg-primary"
                    transition={{ type: "spring", stiffness: 500, damping: 40 }}
                  />
                )}
                <span className={cn("relative z-10", active ? "text-primary-foreground" : "text-text-muted hover:text-text")}>
                  {LEVEL_LABELS[config.level] ?? config.level}
                </span>
              </button>
            );
          })}
        </div>

        <Dropdown
          className="w-auto min-w-[10rem]"
          value={density}
          onChange={onDensityChange}
          ariaLabel="Mine density"
          options={densityOptions.map((d) => ({
            value: d,
            label: `${DENSITY_LABELS[d] ?? d}${activeConfig ? ` (${activeConfig.densities[d]} mines)` : ""}`,
          }))}
        />

        {activeConfig && (
          <span className="font-mono text-xs text-text-muted">
            {activeConfig.rows}x{activeConfig.cols}
          </span>
        )}
      </div>
    </div>
  );
}
