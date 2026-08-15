import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cellKey, computeFrontier, generateBoard, revealCascade, type GeneratedBoard } from "@/lib/generativeBoard";

/**
 * The landing page's fixed backdrop -- a full-viewport, edge-to-edge
 * Minesweeper board rather than a small decorative motif, taking the place
 * of the original design brief's licensed background video. Generated with
 * real adjacency rules (`generativeBoard.ts`) so revealed numbers are
 * mathematically consistent with their hidden mine neighbors.
 *
 * Genuinely never stops: one more frontier cell opens every
 * `REVEAL_TICK_MS`, the way a real game reveals cells one click at a time,
 * until the reachable region is fully solved -- then, after a pause, a fresh
 * board fades in and it starts again. There is no cap that leaves it idle.
 *
 * Cells render with no per-cell border (a full 26x14 grid of bordered tiles
 * reads as graph paper, not a board) -- only a bevel `box-shadow` implies
 * each tile's edge, the way a real Minesweeper board's tiles are implied by
 * light/shadow rather than drawn outlines, so the whole grid reads as one
 * continuous surface.
 */
/** Exported because `Layout`'s vertical guide lines snap themselves onto this
 * grid's seams. If the two ever disagree the guides land mid-tile and read as a
 * second, misaligned set of board lines -- which is exactly what happened while
 * they were positioned at a fixed `36rem` instead. */
export const COLS = 26;
const ROWS = 14;
const MINE_DENSITY = 0.16;
const INITIAL_REVEAL_FRACTION = 0.16;
const REVEAL_TICK_MS = 3000;
const RESTART_PAUSE_MS = 5000;
const RECENT_REVEAL_WINDOW = 6;
const FRONTIER_GLOW_SAMPLE = 5;
const BOARD_FADE_SECONDS = 1.6;

function sample<T>(items: T[], n: number): T[] {
  const copy = [...items];
  const out: T[] = [];
  while (copy.length > 0 && out.length < n) {
    out.push(copy.splice(Math.floor(Math.random() * copy.length), 1)[0]);
  }
  return out;
}

function freshBoardState(board: GeneratedBoard) {
  return {
    revealed: new Set(board.initialRevealed),
    frontier: [...board.initialFrontier],
    recentReveals: board.initialRevealed.slice(-RECENT_REVEAL_WINDOW).reverse(),
    frontierGlow: sample(board.initialFrontier, FRONTIER_GLOW_SAMPLE),
  };
}

function newBoard(): GeneratedBoard {
  return generateBoard(ROWS, COLS, MINE_DENSITY, INITIAL_REVEAL_FRACTION);
}

export function LandingBackground() {
  const [board, setBoard] = useState(newBoard);
  // Only used as the `AnimatePresence`/`motion.div` `key` below, to trigger
  // its cross-fade when a fresh board replaces the old one -- deliberately
  // not a dependency of any hook, just a render identity for React.
  const [boardKey, setBoardKey] = useState(0);

  const initial = useMemo(() => freshBoardState(board), [board]);
  const revealedRef = useRef(initial.revealed);
  const frontierRef = useRef(initial.frontier);

  const [revealed, setRevealed] = useState(initial.revealed);
  const [recentReveals, setRecentReveals] = useState(initial.recentReveals);
  const [frontierGlow, setFrontierGlow] = useState(initial.frontierGlow);

  useEffect(() => {
    revealedRef.current = initial.revealed;
    frontierRef.current = initial.frontier;
    setRevealed(initial.revealed);
    setRecentReveals(initial.recentReveals);
    setFrontierGlow(initial.frontierGlow);
  }, [initial]);

  useEffect(() => {
    let restartTimer: ReturnType<typeof setTimeout> | undefined;

    const id = setInterval(() => {
      if (frontierRef.current.length === 0) {
        if (!restartTimer) {
          restartTimer = setTimeout(() => {
            setBoard(newBoard());
            setBoardKey((k) => k + 1);
          }, RESTART_PAUSE_MS);
        }
        return;
      }

      const index = Math.floor(Math.random() * frontierRef.current.length);
      const [startKey] = frontierRef.current.splice(index, 1);
      const newlyRevealed = revealCascade(board.cells, board.rows, board.cols, startKey, revealedRef.current);
      if (newlyRevealed.length === 0) return;

      const next = new Set(revealedRef.current);
      for (const key of newlyRevealed) next.add(key);
      revealedRef.current = next;
      frontierRef.current = computeFrontier(board.cells, board.rows, board.cols, next);

      setRevealed(next);
      setRecentReveals((prev) => [...newlyRevealed, ...prev].slice(0, RECENT_REVEAL_WINDOW));
      setFrontierGlow(sample(frontierRef.current, FRONTIER_GLOW_SAMPLE));
    }, REVEAL_TICK_MS);

    return () => {
      clearInterval(id);
      if (restartTimer) clearTimeout(restartTimer);
    };
  }, [board]);

  const flaggedSet = useMemo(() => new Set(board.flagged), [board]);
  const pulsingSet = useMemo(() => new Set(recentReveals), [recentReveals]);
  const justChosenKey = recentReveals[0] ?? null;
  const frontierGlowSet = useMemo(() => new Set(frontierGlow), [frontierGlow]);

  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-[#0b0c0e]" aria-hidden="true">
      <div
        className="absolute inset-0"
        style={{
          // `mask-image` is a *luminance* mask, not an alpha one -- white is
          // "fully visible", black is "fully hidden" (the reverse of a plain
          // opacity gradient).
          maskImage: "radial-gradient(ellipse 80% 80% at 50% 38%, white 45%, transparent 95%)",
          WebkitMaskImage: "radial-gradient(ellipse 80% 80% at 50% 38%, white 45%, transparent 95%)",
        }}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={boardKey}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: BOARD_FADE_SECONDS, ease: "easeInOut" }}
            className="grid h-full w-full"
            style={{ gridTemplateColumns: `repeat(${COLS}, 1fr)`, gridTemplateRows: `repeat(${ROWS}, 1fr)` }}
          >
            {board.cells.map((row, r) =>
              row.map((cell, c) => {
                const key = cellKey(r, c);
                const isRevealed = revealed.has(key);
                const isFlagged = !isRevealed && flaggedSet.has(key);
                const isPulsing = isRevealed && pulsingSet.has(key);
                const isJustChosen = isRevealed && key === justChosenKey;
                const isFrontierGlow = !isRevealed && !isFlagged && frontierGlowSet.has(key);

                return (
                  <MinesweeperCell
                    key={key}
                    count={cell.count}
                    isRevealed={isRevealed}
                    isFlagged={isFlagged}
                    isPulsing={isPulsing}
                    isJustChosen={isJustChosen}
                    isFrontierGlow={isFrontierGlow}
                  />
                );
              }),
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      <motion.div
        className="absolute -top-32 left-1/4 h-[28rem] w-[28rem] rounded-full bg-[#00d2ff]/[0.06] blur-[120px]"
        animate={{ opacity: [0.5, 0.9, 0.5], x: [0, 24, 0] }}
        transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute top-1/3 -right-24 h-[24rem] w-[24rem] rounded-full bg-[#3987e5]/[0.06] blur-[120px]"
        animate={{ opacity: [0.4, 0.8, 0.4], y: [0, -28, 0] }}
        transition={{ duration: 24, repeat: Infinity, ease: "easeInOut", delay: 2 }}
      />

      <div className="absolute inset-0 bg-gradient-to-b from-[#0b0c0e]/45 via-transparent to-[#0b0c0e]/60" />
    </div>
  );
}

function MinesweeperCell({
  count,
  isRevealed,
  isFlagged,
  isPulsing,
  isJustChosen,
  isFrontierGlow,
}: {
  count: number;
  isRevealed: boolean;
  isFlagged: boolean;
  isPulsing: boolean;
  isJustChosen: boolean;
  isFrontierGlow: boolean;
}) {
  // No `border` here on purpose -- a border on every one of ~360 cells draws
  // a continuous grid-line lattice across the whole viewport, which reads as
  // graph paper/disconnected panels rather than one board. The bevel
  // (`boxShadow`) alone implies each tile's edge, the way a real
  // Minesweeper board's tiles are, without a drawn seam between every pair.
  const baseStyle: React.CSSProperties = isRevealed
    ? {
        background: "rgba(148, 163, 184, 0.1)",
        boxShadow: "inset 0 1px 3px rgba(0,0,0,0.5)",
      }
    : {
        background: "linear-gradient(135deg, rgba(148,163,184,0.16), rgba(15,23,42,0.26))",
        boxShadow: "inset 1px 1px 0 rgba(255,255,255,0.07), inset -1px -1px 0 rgba(0,0,0,0.35)",
      };

  return (
    <div className="relative flex items-center justify-center" style={baseStyle}>
      {isJustChosen && (
        <motion.div
          className="absolute inset-0"
          initial={{ opacity: 1, scale: 1.15 }}
          animate={{ opacity: 0, scale: 1 }}
          transition={{ duration: 1.4, ease: "easeOut" }}
          style={{ boxShadow: "inset 0 0 0 2px rgba(164,244,253,0.9), 0 0 16px 2px rgba(0,210,255,0.55)" }}
        />
      )}
      {isPulsing && (
        <motion.div
          className="absolute inset-0"
          initial={{ opacity: 0.4 }}
          animate={{ opacity: 0 }}
          transition={{ duration: RECENT_REVEAL_WINDOW * (REVEAL_TICK_MS / 1000), ease: "easeOut" }}
          style={{ boxShadow: "inset 0 0 0 1px rgba(0,210,255,0.3), 0 0 8px rgba(0,210,255,0.15)" }}
        />
      )}
      {isFrontierGlow && (
        <motion.div
          className="absolute inset-0"
          animate={{ opacity: [0.1, 0.28, 0.1] }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
          style={{ boxShadow: "inset 0 0 0 1px rgba(0,210,255,0.28)" }}
        />
      )}
      {isRevealed && count > 0 && (
        <span className="font-mono text-[11px] font-medium" style={{ color: "rgba(255,255,255,0.78)" }}>
          {count}
        </span>
      )}
      {isFlagged && (
        <span
          className="h-1.5 w-1.5 rounded-full"
          style={{ background: "rgba(0,210,255,0.4)", boxShadow: "0 0 6px rgba(0,210,255,0.45)" }}
        />
      )}
    </div>
  );
}
