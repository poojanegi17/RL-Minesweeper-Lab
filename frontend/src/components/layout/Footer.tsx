export function Footer() {
  return (
    <footer className="border-t border-border">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-1 px-4 py-8 text-sm sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <p className="text-text-muted">
          RL Minesweeper Lab — comparing rule-based and reinforcement learning
          approaches to Minesweeper.
        </p>
        <p className="text-xs text-text-muted/70">
          Live results, powered by a FastAPI backend reading directly from the RL training pipeline's own artifacts.
        </p>
      </div>
    </footer>
  );
}
