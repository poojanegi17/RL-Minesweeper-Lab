interface ReplayTimelineProps {
  stepIndex: number;
  totalSteps: number;
  onScrub: (stepIndex: number) => void;
}

/** "Step N / Total" label plus a scrubber for jumping to any point in the episode. */
export function ReplayTimeline({ stepIndex, totalSteps, onScrub }: ReplayTimelineProps) {
  return (
    <div className="flex flex-col gap-2">
      <p className="font-mono text-sm text-text-muted">
        Step {stepIndex} / {totalSteps}
      </p>
      <input
        type="range"
        min={0}
        max={totalSteps}
        value={stepIndex}
        onChange={(e) => onScrub(Number(e.target.value))}
        style={{ accentColor: "var(--color-primary)" }}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-surface-hover"
        aria-label="Replay step"
      />
    </div>
  );
}
