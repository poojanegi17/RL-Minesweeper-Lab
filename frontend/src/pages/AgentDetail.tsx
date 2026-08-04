import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Hourglass } from "lucide-react";
import { Tabs } from "@/components/ui/Tabs";
import { EmptyState } from "@/components/ui/EmptyState";
import { PlaceholderChart } from "@/components/charts/PlaceholderChart";
import { AgentStatusBadge } from "@/components/agent/AgentStatusBadge";
import { AGENT_ICONS } from "@/components/agent/agentIcons";
import { useTheme } from "@/app/ThemeProvider";
import { getAgentById } from "@/data/agents";
import { AGENT_HEX, AGENT_STYLES } from "@/data/types";
import { cn } from "@/lib/cn";

export function AgentDetail() {
  const { agentId } = useParams<{ agentId: string }>();
  const agent = agentId ? getAgentById(agentId) : undefined;
  const { theme } = useTheme();

  if (!agent) {
    return (
      <EmptyState
        icon={Hourglass}
        title="Agent not found"
        description="This agent doesn't exist yet. Check the agent catalog for the current lineup."
      />
    );
  }

  const style = AGENT_STYLES[agent.kind];
  const Icon = AGENT_ICONS[agent.kind];

  return (
    <div className="flex flex-col gap-8">
      <Link
        to="/agents"
        className="flex w-fit items-center gap-1.5 text-sm text-text-muted transition-colors hover:text-text"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to agents
      </Link>

      <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-4">
          <span
            className={cn(
              "flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-current/10",
              style.text,
            )}
          >
            <Icon className="h-6 w-6" />
          </span>
          <div>
            <h1 className="text-3xl font-semibold tracking-tight text-heading">
              {agent.name}
            </h1>
            <p className="mt-1.5 max-w-xl text-text-muted">{agent.tagline}</p>
          </div>
        </div>
        <AgentStatusBadge status={agent.status} />
      </div>

      <Tabs
        tabs={[
          { id: "overview", label: "Overview" },
          { id: "architecture", label: "Architecture" },
          { id: "metrics", label: "Metrics" },
        ]}
      >
        {(active) => (
          <>
            {active === "overview" && (
              <p className="max-w-2xl leading-relaxed text-text">
                {agent.description}
              </p>
            )}
            {active === "architecture" && (
              <p className="max-w-2xl font-mono text-[13px] leading-relaxed text-text">
                {agent.architecture}
              </p>
            )}
            {active === "metrics" &&
              (agent.metrics.length > 0 ? (
                <PlaceholderChart
                  data={agent.metrics.map((metric) => ({
                    label: metric.label,
                    value: metric.value,
                  }))}
                  color={AGENT_HEX[agent.kind][theme]}
                />
              ) : (
                <EmptyState
                  icon={Hourglass}
                  title="Training not yet complete"
                  description="This agent's training run hasn't finished, so metrics aren't available yet."
                />
              ))}
          </>
        )}
      </Tabs>
    </div>
  );
}
