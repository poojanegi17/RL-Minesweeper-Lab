import { mockAgents } from "./mockAgents";
import type { Agent } from "./types";

export function getAgents(): Agent[] {
  return mockAgents;
}

export function getAgentById(id: string): Agent | undefined {
  return mockAgents.find((agent) => agent.id === id);
}
