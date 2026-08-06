import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/cn";

const SPEEDS = [1, 2, 4] as const;

interface ReplayControlsProps {
  stepIndex: number;
  totalSteps: number;
  isPlaying: boolean;
  speed: number;
  onPrevious: () => void;
  onNext: () => void;
  onTogglePlay: () => void;
  onSpeedChange: (speed: number) => void;
}

/** Previous/Play-Pause/Next transport controls plus a playback-speed selector. */
export function ReplayControls({
  stepIndex,
  totalSteps,
  isPlaying,
  speed,
  onPrevious,
  onNext,
  onTogglePlay,
  onSpeedChange,
}: ReplayControlsProps) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-4">
      <div className="flex items-center gap-2">
        <Button variant="secondary" size="sm" onClick={onPrevious} disabled={stepIndex <= 0}>
          <ChevronLeft className="h-4 w-4" />
          Previous
        </Button>
        <Button variant="primary" size="sm" onClick={onTogglePlay} disabled={stepIndex >= totalSteps && !isPlaying}>
          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          {isPlaying ? "Pause" : "Play"}
        </Button>
        <Button variant="secondary" size="sm" onClick={onNext} disabled={stepIndex >= totalSteps}>
          Next
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>

      <div className="flex items-center gap-1.5 text-sm text-text-muted">
        Speed:
        {SPEEDS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => onSpeedChange(s)}
            className={cn(
              "rounded-md px-2 py-1 font-mono text-xs transition-colors",
              speed === s ? "bg-primary text-primary-foreground" : "bg-surface-hover text-text-muted hover:text-text",
            )}
          >
            {s}x
          </button>
        ))}
      </div>
    </div>
  );
}
