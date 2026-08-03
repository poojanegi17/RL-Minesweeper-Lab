import { useSearchParams } from "react-router-dom";
import { Scale } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Select } from "@/components/ui/Select";
import { EmptyState } from "@/components/ui/EmptyState";
import { PlaceholderChart } from "@/components/charts/PlaceholderChart";
import { AgentStatusBadge } from "@/components/agent/AgentStatusBadge";
import { AGENT_ICONS } from "@/components/agent/agentIcons";
import { useTheme } from "@/app/ThemeProvider";
import { getAgentById, getAgents } from "@/data/agents";
import { AGENT_HEX, AGENT_STYLES, type Agent } from "@/data/types";
import { cn } from "@/lib/cn";

export function Compare() {
  const agents = getAgents();
  const { theme } = useTheme();
  const [searchParams, setSearchParams] = useSearchParams();

  const agentAId = searchParams.get("a") ?? agents[0]?.id;
  const agentBId = searchParams.get("b") ?? agents[1]?.id;

  const agentA = getAgentById(agentAId);
  const agentB = getAgentById(agentBId);

  function updateSelection(slot: "a" | "b", value: string) {
    const next = new URLSearchParams(searchParams);
    next.set(slot, value);
    setSearchParams(next);
  }

  const winRateA = agentA?.metrics.find((m) => m.label === "Win Rate");
  const winRateB = agentB?.metrics.find((m) => m.label === "Win Rate");

  return (
    <div className="flex flex-col gap-10">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight text-heading">
          Compare
        </h1>
        <p className="mt-2 max-w-2xl text-text-muted">
          Pick two agents to compare their win rate and headline metrics
          side by side.
        </p>
      </div>

      <div className="flex flex-col items-center gap-4 sm:flex-row">
        <label className="flex w-full flex-col gap-1.5 text-sm text-text-muted">
          Agent A
          <Select
            value={agentAId}
            onChange={(e) => updateSelection("a", e.target.value)}
          >
            {agents.map((agent) => (
              <option key={agent.id} value={agent.id}>
                {agent.name}
              </option>
            ))}
          </Select>
        </label>
        <span className="font-mono text-xs text-text-muted sm:mt-6">vs</span>
        <label className="flex w-full flex-col gap-1.5 text-sm text-text-muted">
          Agent B
          <Select
            value={agentBId}
            onChange={(e) => updateSelection("b", e.target.value)}
          >
            {agents.map((agent) => (
              <option key={agent.id} value={agent.id}>
                {agent.name}
              </option>
            ))}
          </Select>
        </label>
      </div>

      {agentA && agentB && winRateA && winRateB ? (
        <Card>
          <h2 className="mb-4 text-sm font-medium tracking-wide text-text-muted uppercase">
            Win rate comparison
          </h2>
          <PlaceholderChart
            data={[
              {
                label: agentA.name,
                value: winRateA.value,
                color: AGENT_HEX[agentA.kind][theme],
              },
              {
                label: agentB.name,
                value: winRateB.value,
                color: AGENT_HEX[agentB.kind][theme],
              },
            ]}
          />
        </Card>
      ) : (
        <EmptyState
          icon={Scale}
          title="No comparable metrics yet"
          description="One or both selected agents haven't finished training, so there's no win rate to compare yet."
        />
      )}

      <div className="grid gap-4 sm:grid-cols-2">
        <AgentSummaryCard agent={agentA} opponent={agentB} />
        <AgentSummaryCard agent={agentB} opponent={agentA} />
      </div>
    </div>
  );
}

interface AgentSummaryCardProps {
  agent: Agent | undefined;
  opponent: Agent | undefined;
}

function AgentSummaryCard({ agent, opponent }: AgentSummaryCardProps) {
  if (!agent) {
    return (
      <Card className="text-sm text-text-muted">Select an agent to compare.</Card>
    );
  }

  const style = AGENT_STYLES[agent.kind];
  const Icon = AGENT_ICONS[agent.kind];

  return (
    <Card className="flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <span
          className={cn(
            "flex h-8 w-8 items-center justify-center rounded-lg bg-current/10",
            style.text,
          )}
        >
          <Icon className="h-4 w-4" />
        </span>
        <h3 className="font-semibold text-heading">{agent.name}</h3>
        <AgentStatusBadge status={agent.status} />
      </div>
      {agent.metrics.length > 0 ? (
        <dl className="flex flex-col gap-2">
          {agent.metrics.map((metric) => {
            const opponentMetric = opponent?.metrics.find(
              (m) => m.label === metric.label,
            );
            const isLeading =
              opponentMetric !== undefined && metric.value > opponentMetric.value;

            return (
              <div
                key={metric.label}
                className="flex items-center justify-between text-sm"
              >
                <dt className="text-text-muted">{metric.label}</dt>
                <dd
                  className={cn(
                    "font-mono font-medium",
                    isLeading ? "text-primary" : "text-text",
                  )}
                >
                  {metric.value.toLocaleString()}
                  {metric.unit}
                </dd>
              </div>
            );
          })}
        </dl>
      ) : (
        <p className="text-sm text-text-muted">No metrics available yet.</p>
      )}
    </Card>
  );
}
