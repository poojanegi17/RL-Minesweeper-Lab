import { motion } from "framer-motion";
import { AgentCard } from "@/components/agent/AgentCard";
import { getAgents } from "@/data/agents";

export function Agents() {
  const agents = getAgents();

  return (
    <div className="flex flex-col gap-10">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight text-heading">
          Agents
        </h1>
        <p className="mt-2 max-w-2xl text-text-muted">
          Four approaches to the same problem, from deterministic logic to
          learned policies. Select one to see how it reasons about the board.
        </p>
      </div>

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
    </div>
  );
}
