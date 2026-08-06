import { Link } from "react-router-dom";
import { ArrowUpRight } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import { AgentStatusBadge } from "@/components/agent/AgentStatusBadge";
import { AGENT_ICONS } from "@/components/agent/agentIcons";
import { AGENT_STYLES, type Agent } from "@/data/types";
import { cn } from "@/lib/cn";

interface AgentCardProps {
  agent: Agent;
}

export function AgentCard({ agent }: AgentCardProps) {
  const style = AGENT_STYLES[agent.kind];
  const Icon = AGENT_ICONS[agent.kind];
  const headline = agent.metrics[0];

  return (
    <Link to={`/agents/${agent.id}`} className="group block h-full">
      <Card interactive className="flex h-full flex-col gap-4">
        <div className="flex items-start justify-between">
          <span
            className={cn(
              "flex h-9 w-9 items-center justify-center rounded-lg bg-current/10",
              style.text,
            )}
          >
            <Icon className="h-4 w-4" />
          </span>
          <div className="flex items-center gap-1.5">
            {agent.type && <Badge variant="outline">{agent.type.replace(/_/g, " ")}</Badge>}
            <AgentStatusBadge status={agent.status} />
          </div>
        </div>

        <div className="flex-1">
          <h3 className="flex items-center gap-1 font-semibold text-heading">
            {agent.name}
            <ArrowUpRight className="h-3.5 w-3.5 text-text-muted opacity-0 transition-opacity group-hover:opacity-100" />
          </h3>
          <p className="mt-1 text-sm text-text-muted">{agent.tagline}</p>
        </div>

        <div className="flex items-center justify-between border-t border-border pt-3 text-xs">
          <span className="text-text-muted">
            {headline ? headline.label : "No metrics yet"}
          </span>
          {headline && (
            <span className="font-mono font-medium text-heading">
              {headline.value.toLocaleString()}
              {headline.unit}
            </span>
          )}
        </div>
      </Card>
    </Link>
  );
}
