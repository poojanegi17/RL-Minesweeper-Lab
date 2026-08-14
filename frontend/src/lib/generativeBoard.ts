/**
 * A synthetic, self-consistent Minesweeper board for the landing page's
 * ambient background (`LandingBackground`) -- purely decorative, generated
 * client-side with real Minesweeper adjacency rules (not arbitrary numbers
 * scattered on a grid) so the revealed clusters read as mathematically
 * coherent rather than random noise: every revealed number genuinely counts
 * its hidden mine neighbors, and a 0 always cascades its neighbors open the
 * way a real flood-fill reveal does.
 */

export interface GeneratedCell {
  mine: boolean;
  count: number;
}

export interface GeneratedBoard {
  rows: number;
  cols: number;
  cells: GeneratedCell[][];
  /** Cell keys ("r,c") revealed by the initial flood-fill, in reveal order --
   * `LandingBackground` seeds its `revealed` state from this and continues
   * revealing further frontier cells over time from where this left off. */
  initialRevealed: string[];
  /** Hidden, non-mine cells bordering the initially revealed region -- the
   * starting queue for the slow "still solving" progressive reveal. */
  initialFrontier: string[];
  /** A handful of mines adjacent to the revealed region, flagged rather than
   * detonated -- the "subtle mine indicator" the design calls for, without
   * ever showing an exploded/lost board. */
  flagged: string[];
}

export function cellKey(row: number, col: number): string {
  return `${row},${col}`;
}

function neighborsOf(row: number, col: number, rows: number, cols: number): [number, number][] {
  const out: [number, number][] = [];
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      const nr = row + dr;
      const nc = col + dc;
      if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) out.push([nr, nc]);
    }
  }
  return out;
}

/**
 * Generates one board and flood-fills a plausible "mid-game" revealed
 * region -- capped well under the full board (`revealCapFraction`) so most
 * of the grid stays unrevealed, matching a real in-progress game rather
 * than a finished one.
 */
export function generateBoard(
  rows: number,
  cols: number,
  mineDensity: number,
  revealCapFraction: number,
): GeneratedBoard {
  const cells: GeneratedCell[][] = Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => ({ mine: false, count: 0 })),
  );

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      cells[r][c].mine = Math.random() < mineDensity;
    }
  }
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (cells[r][c].mine) continue;
      cells[r][c].count = neighborsOf(r, c, rows, cols).filter(([nr, nc]) => cells[nr][nc].mine).length;
    }
  }

  const totalNonMineCells = cells.flat().filter((cell) => !cell.mine).length;
  const revealCap = Math.floor(totalNonMineCells * revealCapFraction);

  const revealed = new Set<string>();
  const revealedOrder: string[] = [];
  const zeroCells: [number, number][] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (!cells[r][c].mine && cells[r][c].count === 0) zeroCells.push([r, c]);
    }
  }

  const seeds = zeroCells.length > 0 ? shuffle(zeroCells).slice(0, 3) : [];
  const queue: [number, number][] = [...seeds];

  while (queue.length > 0 && revealed.size < revealCap) {
    const [r, c] = queue.shift()!;
    const key = cellKey(r, c);
    if (revealed.has(key) || cells[r][c].mine) continue;
    revealed.add(key);
    revealedOrder.push(key);
    if (cells[r][c].count === 0) {
      for (const [nr, nc] of neighborsOf(r, c, rows, cols)) {
        if (!revealed.has(cellKey(nr, nc)) && !cells[nr][nc].mine) queue.push([nr, nc]);
      }
    }
  }

  // If the board rolled few/no zero cells (unlucky mine placement), fall
  // back to revealing a handful of the lowest-count cells so there's still
  // something on screen rather than an all-hidden board.
  if (revealed.size === 0) {
    const byCount = cells
      .flat()
      .map((cell, i) => ({ cell, r: Math.floor(i / cols), c: i % cols }))
      .filter((entry) => !entry.cell.mine)
      .sort((a, b) => a.cell.count - b.cell.count)
      .slice(0, Math.min(revealCap, 12));
    for (const { r, c } of byCount) {
      const key = cellKey(r, c);
      revealed.add(key);
      revealedOrder.push(key);
    }
  }

  const frontierSet = new Set<string>();
  for (const key of revealed) {
    const [r, c] = key.split(",").map(Number);
    for (const [nr, nc] of neighborsOf(r, c, rows, cols)) {
      const nKey = cellKey(nr, nc);
      if (!revealed.has(nKey) && !cells[nr][nc].mine) frontierSet.add(nKey);
    }
  }

  const flaggableMines: string[] = [];
  for (const key of revealed) {
    const [r, c] = key.split(",").map(Number);
    for (const [nr, nc] of neighborsOf(r, c, rows, cols)) {
      if (cells[nr][nc].mine) flaggableMines.push(cellKey(nr, nc));
    }
  }
  const flagged = shuffle([...new Set(flaggableMines)]).slice(0, 4);

  return {
    rows,
    cols,
    cells,
    initialRevealed: revealedOrder,
    initialFrontier: shuffle([...frontierSet]),
    flagged,
  };
}

/**
 * Reveals `startKey` and cascades through any connected zero-count region,
 * exactly like clicking a real Minesweeper cell -- used by
 * `LandingBackground`'s slow progressive-reveal timer to open one more
 * frontier cell at a time without re-running the whole board generator.
 * Returns the newly revealed keys (doesn't mutate `alreadyRevealed`, so the
 * caller decides how/when to merge them into its own state).
 */
export function revealCascade(
  cells: GeneratedCell[][],
  rows: number,
  cols: number,
  startKey: string,
  alreadyRevealed: ReadonlySet<string>,
): string[] {
  const newlyRevealed = new Set<string>();
  const queue: string[] = [startKey];

  while (queue.length > 0) {
    const key = queue.shift()!;
    if (alreadyRevealed.has(key) || newlyRevealed.has(key)) continue;
    const [r, c] = key.split(",").map(Number);
    if (cells[r][c].mine) continue;
    newlyRevealed.add(key);
    if (cells[r][c].count === 0) {
      for (const [nr, nc] of neighborsOf(r, c, rows, cols)) {
        const nKey = cellKey(nr, nc);
        if (!alreadyRevealed.has(nKey) && !newlyRevealed.has(nKey)) queue.push(nKey);
      }
    }
  }

  return [...newlyRevealed];
}

/** Every hidden, non-mine cell bordering `revealed` -- recomputed after each
 * progressive-reveal tick rather than incrementally patched, since the
 * board is small enough that a full scan is cheap and this stays correct
 * even when a cascade reveals several cells in one tick. */
export function computeFrontier(
  cells: GeneratedCell[][],
  rows: number,
  cols: number,
  revealed: ReadonlySet<string>,
): string[] {
  const frontier = new Set<string>();
  for (const key of revealed) {
    const [r, c] = key.split(",").map(Number);
    for (const [nr, nc] of neighborsOf(r, c, rows, cols)) {
      const nKey = cellKey(nr, nc);
      if (!revealed.has(nKey) && !cells[nr][nc].mine) frontier.add(nKey);
    }
  }
  return [...frontier];
}

function shuffle<T>(items: T[]): T[] {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}
