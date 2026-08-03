import { Badge } from "@/components/ui/Badge";
import type { AgentStatus } from "@/data/types";
import { cn } from "@/lib/cn";

const STATUS_LABEL: Record<AgentStatus, string> = {
  baseline: "Baseline",
  trained: "Trained",
  planned: "Planned",
};

const STATUS_DOT: Record<AgentStatus, string> = {
  baseline: "bg-text-muted",
  trained: "bg-emerald-500",
  planned: "bg-primary animate-pulse",
};

interface AgentStatusBadgeProps {
  status: AgentStatus;
}

export function AgentStatusBadge({ status }: AgentStatusBadgeProps) {
  return (
    <Badge variant="outline">
      <span className={cn("h-1.5 w-1.5 rounded-full", STATUS_DOT[status])} />
      {STATUS_LABEL[status]}
    </Badge>
  );
}
