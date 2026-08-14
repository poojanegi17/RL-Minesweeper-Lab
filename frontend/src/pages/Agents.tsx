import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Users } from "lucide-react";
import { AgentCard } from "@/components/agent/AgentCard";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { getAgents } from "@/api/agents";
import { getLeaderboard } from "@/api/metrics";
import { useApiQuery } from "@/hooks/useApiQuery";
import { toUiAgent } from "@/lib/agentAdapters";

// Fetched together so each card can show its headline win rate without a
// second round-trip per agent -- /api/leaderboard already computes it.
async function fetchAgentCatalog() {
  const [agents, leaderboard] = await Promise.all([getAgents(), getLeaderboard()]);
  return agents.map((agent) =>
    toUiAgent(
      agent,
      leaderboard.find((entry) => entry.agent === agent.name),
    ),
  );
}

export function Agents() {
  const { data: agents, status, error, isSlow, retry } = useApiQuery(fetchAgentCatalog, []);

  return (
    <div className="flex flex-col gap-10">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight text-heading">
          Agents
        </h1>
        <p className="mt-2 max-w-2xl text-text-muted">
          Five approaches to the same problem, from a random baseline to
          learned policies. Select one to see how it reasons about the
          board. Only the best-performing agent at each config is present —
          to see all of them, explore the{" "}
          <Link to="/research" className="text-primary hover:underline">
            research pipeline
          </Link>
          .
        </p>
      </div>

      {status === "loading" && (
        <div className="flex flex-col gap-4">
          <div className="grid gap-4 sm:grid-cols-2" aria-label="Loading agents">
            {Array.from({ length: 4 }).map((_, index) => (
              <Skeleton key={index} className="h-40" />
            ))}
          </div>
          {isSlow && <ColdStartNotice />}
        </div>
      )}

      {status === "error" && error && (
        <ApiErrorState error={error} onRetry={retry} title="Couldn't load agents" />
      )}

      {status === "success" && agents && agents.length === 0 && (
        <EmptyState
          icon={Users}
          title="No agents found"
          description="The backend's agent catalog is empty."
        />
      )}

      {status === "success" && agents && agents.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2">
          {agents.map((agent, index) => (
            <motion.div
              key={agent.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, delay: index * 0.06, ease: "easeOut" }}
            >
              <AgentCard agent={agent} />
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
