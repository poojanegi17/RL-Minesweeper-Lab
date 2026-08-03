export type AgentKind = "rule-based" | "q-learning" | "dqn" | "ppo";

export type AgentStatus = "baseline" | "trained" | "planned";

export interface Metric {
  label: string;
  value: number;
  unit?: string;
}

export interface Agent {
  id: string;
  kind: AgentKind;
  name: string;
  tagline: string;
  description: string;
  architecture: string;
  status: AgentStatus;
  metrics: Metric[];
}

interface AgentStyle {
  text: string;
  bg: string;
  border: string;
  dot: string;
}

// Mirrors the --color-agent-* tokens in index.css. Recharts needs a literal
// hex (SVG fill can't resolve CSS custom properties reliably), and the two
// modes' steps differ, so callers pick a side via the active theme:
// AGENT_HEX[agent.kind][theme]. Validated as a set with the dataviz skill's
// CVD/contrast checker — don't hand-tune without re-running it.
export const AGENT_HEX: Record<AgentKind, { light: string; dark: string }> = {
  "rule-based": { light: "#008300", dark: "#008300" },
  "q-learning": { light: "#2a78d6", dark: "#3987e5" },
  dqn: { light: "#e87ba4", dark: "#d55181" },
  ppo: { light: "#eda100", dark: "#c98500" },
};

export const AGENT_STYLES: Record<AgentKind, AgentStyle> = {
  "rule-based": {
    text: "text-agent-rule-based",
    bg: "bg-agent-rule-based",
    border: "border-agent-rule-based",
    dot: "bg-agent-rule-based",
  },
  "q-learning": {
    text: "text-agent-q-learning",
    bg: "bg-agent-q-learning",
    border: "border-agent-q-learning",
    dot: "bg-agent-q-learning",
  },
  dqn: {
    text: "text-agent-dqn",
    bg: "bg-agent-dqn",
    border: "border-agent-dqn",
    dot: "bg-agent-dqn",
  },
  ppo: {
    text: "text-agent-ppo",
    bg: "bg-agent-ppo",
    border: "border-agent-ppo",
    dot: "bg-agent-ppo",
  },
};
