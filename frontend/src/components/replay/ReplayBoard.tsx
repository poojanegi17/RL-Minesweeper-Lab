import { Bomb } from "lucide-react";
import { NUMBER_COLORS } from "@/components/board/BoardIllustration";
import { cn } from "@/lib/cn";

interface HighlightedCell {
  row: number;
  col: number;
}

interface ReplayBoardProps {
  /** -1 = hidden, 0-8 = revealed adjacent-mine count. Exactly the observation
   * the agent received -- see `ReplayStep.board_state` / the README's Replay
   * Visualization section for why nothing here ever goes beyond that. */
  board: number[][];
  /** The action taken on the step currently being shown, if any (no
   * highlight at "step 0", before any action). */
  highlightedCell?: HighlightedCell | null;
  /**
   * True only when `highlightedCell` is the fatal move of a lost episode
   * (`done && !won` on that step) -- passed in by the caller, derived from
   * data the episode already legitimately reported, never read off the
   * board itself. The engine's own board array shows a hit mine as a plain
   * "0" (see `rl/environment/minesweeper.py`'s `_compute_adjacent_counts`,
   * which never assigns mine cells a count), so without this flag the
   * viewer would otherwise misrepresent the one cell everyone already knows
   * was a mine as an ordinary safe reveal.
   */
  mineHit?: boolean;
}

/**
 * Renders a real, data-driven Minesweeper board for the replay viewer.
 * Reuses `BoardIllustration`'s clue-number color map (`NUMBER_COLORS`)
 * rather than a separately invented palette -- `BoardIllustration` itself is
 * a fixed, non-parameterized decorative component and can't render arbitrary
 * board data, so this is the one real board-rendering component in the app.
 *
 * Tile colors are literal (not the site's `--color-surface`/`--color-primary`
 * theme tokens) to match `LandingBackground`'s dark bevel/neon-blue tile
 * treatment everywhere this board appears -- the whole site is one fixed
 * dark aesthetic now (see `ThemeProvider`), so a real board and the ambient
 * decorative one should look like the same game, not two different themes.
 */
export function ReplayBoard({ board, highlightedCell, mineHit = false }: ReplayBoardProps) {
  const cols = board[0]?.length ?? 0;

  return (
    <div className="inline-block rounded-xl border border-white/10 bg-black/30 p-3 backdrop-blur-md">
      <div
        className="inline-grid gap-[3px]"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
      >
        {board.flatMap((row, r) =>
          row.map((value, c) => {
            const isHighlighted = highlightedCell?.row === r && highlightedCell?.col === c;
            const isHiddenCell = value === -1;
            const isMineHitHere = isHighlighted && mineHit;

            return (
              <div
                key={`${r}-${c}`}
                className={cn(
                  "flex h-10 w-10 items-center justify-center rounded-[5px] font-mono text-sm font-semibold transition-colors",
                  isHiddenCell &&
                    !isMineHitHere &&
                    "border border-white/[0.07] bg-[linear-gradient(135deg,rgba(148,163,184,0.20),rgba(15,23,42,0.32))] shadow-[inset_1px_1px_0_rgba(255,255,255,0.09),inset_-1px_-1px_0_rgba(0,0,0,0.4)]",
                  !isHiddenCell && !isMineHitHere && "border border-white/[0.06] bg-white/[0.06] shadow-[inset_0_1px_3px_rgba(0,0,0,0.5)]",
                  isMineHitHere && "border border-red-400/50 bg-red-500/10",
                  isHighlighted &&
                    !isMineHitHere &&
                    "ring-2 ring-[#00d2ff] ring-offset-1 ring-offset-black shadow-[0_0_10px_rgba(0,210,255,0.35)]",
                )}
              >
                {isMineHitHere && <Bomb className="h-4 w-4 text-red-400" />}
                {!isMineHitHere && !isHiddenCell && value > 0 && (
                  <span className={NUMBER_COLORS[value]}>{value}</span>
                )}
              </div>
            );
          }),
        )}
      </div>
    </div>
  );
}
