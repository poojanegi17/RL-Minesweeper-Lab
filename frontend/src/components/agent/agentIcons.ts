import { Brain, Dices, GitBranch, Grid3x3, Sparkles, type LucideIcon } from "lucide-react";
import type { AgentKind } from "@/data/types";

export const AGENT_ICONS: Record<AgentKind, LucideIcon> = {
  random: Dices,
  "rule-based": GitBranch,
  "q-learning": Grid3x3,
  dqn: Brain,
  ppo: Sparkles,
};
