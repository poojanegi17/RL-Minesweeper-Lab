/** Mirrors `backend/app/schemas/board_config.py`. */

export interface BoardLevelInfo {
  level: string;
  rows: number;
  cols: number;
  /** Density name -> mine count, e.g. { sparse: 3, standard: 5, dense: 8 }. */
  densities: Record<string, number>;
}
