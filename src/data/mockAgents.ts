import type { Agent } from "./types";

export const mockAgents: Agent[] = [
  {
    id: "rule-based",
    kind: "rule-based",
    name: "Rule-Based Solver",
    tagline: "Deterministic logic, no learning involved",
    description:
      "Applies classic Minesweeper deduction rules — flagging cells that must be mines and revealing cells that must be safe — using only the numbers already visible on the board. Serves as the baseline every learning agent is measured against.",
    architecture:
      "Constraint-propagation solver: for each revealed number, compares adjacent unrevealed cells against the count of adjacent flags to deduce guaranteed-safe or guaranteed-mine cells. Falls back to random choice among remaining cells when no deduction is possible.",
    status: "baseline",
    metrics: [
      { label: "Win Rate", value: 62, unit: "%" },
      { label: "Avg. Moves / Game", value: 34 },
      { label: "Boards Solved", value: 6200 },
    ],
  },
  {
    id: "q-learning",
    kind: "q-learning",
    name: "Tabular Q-Learning",
    tagline: "Learns a state-action value table through trial and error",
    description:
      "Maintains a table of expected rewards for each (state, action) pair on small boards, updating values after every move based on observed outcomes. Demonstrates classic reinforcement learning without any neural network.",
    architecture:
      "Q-table indexed by a discretized local-neighborhood state representation, updated with the standard Q-learning update rule (reward plus discounted max future value). Epsilon-greedy exploration during training, greedy action selection at inference.",
    status: "trained",
    metrics: [
      { label: "Win Rate", value: 71, unit: "%" },
      { label: "Episodes Trained", value: 50000 },
      { label: "Avg. Reward", value: 8.4 },
    ],
  },
  {
    id: "dqn",
    kind: "dqn",
    name: "Deep Q-Network",
    tagline: "A convolutional network approximates Q-values across the full board",
    description:
      "Replaces the Q-table with a convolutional neural network so the agent can generalize across board sizes and states it has never seen exactly before, rather than memorizing individual states.",
    architecture:
      "CNN encoder over the board grid feeding a dense head that outputs one Q-value per cell. Trained with experience replay and a target network for stability, following the standard DQN recipe.",
    status: "trained",
    metrics: [
      { label: "Win Rate", value: 79, unit: "%" },
      { label: "Episodes Trained", value: 200000 },
      { label: "Avg. Reward", value: 11.2 },
    ],
  },
  {
    id: "ppo",
    kind: "ppo",
    name: "PPO",
    tagline: "Policy-gradient agent, training in progress",
    description:
      "Learns a stochastic policy directly rather than an action-value function, optimized with Proximal Policy Optimization for stable updates. Included to round out the comparison between value-based and policy-based methods.",
    architecture:
      "Actor-critic network sharing a convolutional trunk, with clipped surrogate objective for the policy update. Training and evaluation are not yet complete — this entry documents the planned setup.",
    status: "planned",
    metrics: [],
  },
];
