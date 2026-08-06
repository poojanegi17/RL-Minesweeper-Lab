import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Bomb, Flag, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { NUMBER_COLORS } from "@/components/board/BoardIllustration";
import { cn } from "@/lib/cn";

// Matches the real project's benchmark board (`backend/app/config.py`'s
// BENCHMARK_BOARD_ROWS/COLS/MINES) -- the same size every agent on this site
// actually trains and plays on, not an arbitrarily chosen demo size.
const ROWS = 5;
const COLS = 5;
const MINES = 5;

type CellState = "hidden" | "revealed" | "flagged";

interface Cell {
  isMine: boolean;
  adjacent: number;
  state: CellState;
}

type GameStatus = "idle" | "playing" | "won" | "lost";

export interface PlayableMinesweeperSummary {
  status: GameStatus;
  revealedCount: number;
  flagCount: number;
}

interface PlayableMinesweeperProps {
  /** Reports a summary up whenever the board changes -- lets the page pair
   * this with the AI comparison panel ("Your strategy" vs. "Agent
   * strategy") without this component knowing anything about that pairing. */
  onStateChange?: (summary: PlayableMinesweeperSummary) => void;
}

function createEmptyBoard(): Cell[][] {
  return Array.from({ length: ROWS }, () => Array.from({ length: COLS }, () => ({ isMine: false, adjacent: 0, state: "hidden" as CellState })));
}

function cloneBoard(board: Cell[][]): Cell[][] {
  return board.map((row) => row.map((cell) => ({ ...cell })));
}

function countAdjacentMines(board: Cell[][], row: number, col: number): number {
  let count = 0;
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      const nr = row + dr;
      const nc = col + dc;
      if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && board[nr][nc].isMine) count++;
    }
  }
  return count;
}

/** Places mines after the first click, excluding that cell -- the first move
 * is always safe, a standard Minesweeper convention (not a claim about the
 * RL environment's own mine placement, which is a separate, backend-owned
 * concern this frontend-only demo doesn't touch). */
function placeMines(board: Cell[][], excludeRow: number, excludeCol: number): Cell[][] {
  const next = cloneBoard(board);
  let placed = 0;
  while (placed < MINES) {
    const r = Math.floor(Math.random() * ROWS);
    const c = Math.floor(Math.random() * COLS);
    if ((r === excludeRow && c === excludeCol) || next[r][c].isMine) continue;
    next[r][c].isMine = true;
    placed++;
  }
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (!next[r][c].isMine) next[r][c].adjacent = countAdjacentMines(next, r, c);
    }
  }
  return next;
}

/** Flood-fills outward from `(row, col)` through connected zero-adjacent
 * cells, the standard Minesweeper cascade-reveal behavior. */
function revealCell(board: Cell[][], row: number, col: number): Cell[][] {
  const next = cloneBoard(board);
  const stack: Array<[number, number]> = [[row, col]];

  while (stack.length > 0) {
    const [r, c] = stack.pop()!;
    if (r < 0 || r >= ROWS || c < 0 || c >= COLS) continue;
    const cell = next[r][c];
    if (cell.state !== "hidden") continue;
    cell.state = "revealed";
    if (cell.adjacent === 0 && !cell.isMine) {
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue;
          stack.push([r + dr, c + dc]);
        }
      }
    }
  }
  return next;
}

function revealAllMines(board: Cell[][]): Cell[][] {
  return board.map((row) => row.map((cell) => (cell.isMine ? { ...cell, state: "revealed" as CellState } : cell)));
}

function checkWin(board: Cell[][]): boolean {
  return board.every((row) => row.every((cell) => cell.isMine || cell.state === "revealed"));
}

function summarize(board: Cell[][], status: GameStatus): PlayableMinesweeperSummary {
  let revealedCount = 0;
  let flagCount = 0;
  for (const row of board) {
    for (const cell of row) {
      if (cell.state === "revealed" && !cell.isMine) revealedCount++;
      if (cell.state === "flagged") flagCount++;
    }
  }
  return { status, revealedCount, flagCount };
}

/**
 * A real, playable, frontend-only Minesweeper board -- no backend or RL
 * environment involved. Lets a visitor feel the problem (partial
 * observability, cascading information, one wrong click ends it) before
 * watching how an agent approaches the exact same decisions.
 */
export function PlayableMinesweeper({ onStateChange }: PlayableMinesweeperProps) {
  const [board, setBoard] = useState<Cell[][]>(createEmptyBoard);
  const [status, setStatus] = useState<GameStatus>("idle");
  const [flagMode, setFlagMode] = useState(false);

  useEffect(() => {
    onStateChange?.(summarize(board, status));
    // `onStateChange` is a caller-provided callback, not a reactive value --
    // including it would re-fire this effect every render if the caller
    // doesn't memoize it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [board, status]);

  function handleReveal(row: number, col: number) {
    if (status === "won" || status === "lost") return;
    const cell = board[row][col];
    if (cell.state === "flagged" || cell.state === "revealed") return;

    let working = board;
    if (status === "idle") {
      working = placeMines(board, row, col);
    }

    const revealed = revealCell(working, row, col);
    if (revealed[row][col].isMine) {
      setBoard(revealAllMines(revealed));
      setStatus("lost");
      return;
    }

    setBoard(revealed);
    setStatus(checkWin(revealed) ? "won" : "playing");
  }

  function handleToggleFlag(row: number, col: number) {
    if (status === "won" || status === "lost") return;
    const cell = board[row][col];
    if (cell.state === "revealed") return;
    setBoard((prev) =>
      prev.map((r, ri) =>
        r.map((c, ci) => (ri === row && ci === col ? { ...c, state: c.state === "flagged" ? "hidden" : "flagged" } : c)),
      ),
    );
  }

  function handleCellClick(row: number, col: number) {
    if (flagMode) handleToggleFlag(row, col);
    else handleReveal(row, col);
  }

  function handleRestart() {
    setBoard(createEmptyBoard());
    setStatus("idle");
  }

  const flagCount = board.flat().filter((c) => c.state === "flagged").length;
  const minesLeft = MINES - flagCount;

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="flex w-full max-w-xs items-center justify-between gap-3 text-sm">
        <Badge variant="outline" className="font-mono">
          {minesLeft} mines left
        </Badge>
        {status === "won" && (
          <motion.div initial={{ scale: 0.7, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: "spring", stiffness: 400, damping: 20 }}>
            <Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400">You win!</Badge>
          </motion.div>
        )}
        {status === "lost" && (
          <motion.div initial={{ scale: 0.7, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: "spring", stiffness: 400, damping: 20 }}>
            <Badge className="bg-red-500/15 text-red-600 dark:text-red-400">Game over</Badge>
          </motion.div>
        )}
        {(status === "idle" || status === "playing") && <span className="text-text-muted">{status === "idle" ? "Click to begin" : "In progress"}</span>}
      </div>

      <div className="relative">
        <motion.div
          className={cn("absolute -inset-4 -z-10 rounded-3xl blur-2xl", status === "won" ? "bg-emerald-500/25" : status === "lost" ? "bg-red-500/20" : "bg-transparent")}
          animate={{ opacity: status === "won" || status === "lost" ? [0.4, 0.7, 0.4] : 0 }}
          transition={{ duration: 1.4, repeat: status === "won" || status === "lost" ? 2 : 0, ease: "easeInOut" }}
          aria-hidden="true"
        />
        <div className="inline-block rounded-xl border border-border bg-surface p-3 shadow-sm shadow-black/[0.02]">
          <div className="inline-grid gap-[3px]" style={{ gridTemplateColumns: `repeat(${COLS}, minmax(0, 1fr))` }}>
            {board.flatMap((row, r) =>
              row.map((cell, c) => {
                const showMine = status === "lost" && cell.isMine && cell.state === "revealed";
                return (
                  <motion.button
                    key={`${r}-${c}-${cell.state}`}
                    type="button"
                    onClick={() => handleCellClick(r, c)}
                    onContextMenu={(e) => {
                      e.preventDefault();
                      handleToggleFlag(r, c);
                    }}
                    disabled={status === "won" || status === "lost"}
                    aria-label={`Cell ${r}, ${c}`}
                    initial={{ scale: 0.7, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    whileHover={cell.state === "hidden" && status !== "won" && status !== "lost" ? { scale: 1.06 } : undefined}
                    whileTap={cell.state === "hidden" && status !== "won" && status !== "lost" ? { scale: 0.94 } : undefined}
                    transition={{ duration: 0.22, ease: "easeOut", delay: showMine ? (r * COLS + c) * 0.035 : 0 }}
                    className={cn(
                      "flex h-11 w-11 items-center justify-center rounded-[5px] font-mono text-sm font-semibold transition-colors",
                      cell.state === "hidden" &&
                        "bg-border/70 shadow-[inset_1px_1px_0_rgba(255,255,255,0.25),inset_-1px_-1px_0_rgba(0,0,0,0.12)] hover:brightness-95 dark:shadow-[inset_1px_1px_0_rgba(255,255,255,0.06),inset_-1px_-1px_0_rgba(0,0,0,0.3)] dark:hover:brightness-125",
                      cell.state === "flagged" &&
                        "bg-border/70 shadow-[inset_1px_1px_0_rgba(255,255,255,0.25),inset_-1px_-1px_0_rgba(0,0,0,0.12)] dark:shadow-[inset_1px_1px_0_rgba(255,255,255,0.06),inset_-1px_-1px_0_rgba(0,0,0,0.3)]",
                      cell.state === "revealed" && !showMine && "border border-border/80 bg-background",
                      showMine && "border border-red-500/50 bg-red-500/10",
                    )}
                  >
                    {cell.state === "flagged" && <Flag className="h-3.5 w-3.5 fill-red-500/20 text-red-500" />}
                    {showMine && <Bomb className="h-4 w-4 text-red-500" />}
                    {cell.state === "revealed" && !cell.isMine && cell.adjacent > 0 && (
                      <span className={NUMBER_COLORS[cell.adjacent]}>{cell.adjacent}</span>
                    )}
                  </motion.button>
                );
              }),
            )}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <Button variant="secondary" size="sm" onClick={() => setFlagMode((v) => !v)} aria-pressed={flagMode}>
          <Flag className="h-3.5 w-3.5" />
          {flagMode ? "Flag mode on" : "Flag mode off"}
        </Button>
        <Button variant="secondary" size="sm" onClick={handleRestart}>
          <RotateCcw className="h-3.5 w-3.5" />
          Restart
        </Button>
      </div>
    </div>
  );
}
