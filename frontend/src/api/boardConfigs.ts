import { apiGet } from "@/api/client";
import type { BoardLevelInfo } from "@/types/boardConfig";

/** GET /api/board-configs -- every difficulty level and its mine-density presets. */
export function getBoardConfigs(): Promise<BoardLevelInfo[]> {
  return apiGet<BoardLevelInfo[]>("/api/board-configs");
}
