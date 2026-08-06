import {
  Coins,
  Cpu,
  Eye,
  GitMerge,
  Grid3x3,
  Layers,
  ListChecks,
  RefreshCw,
  Scale,
  Target,
  type LucideIcon,
} from "lucide-react";
import type { AgentKind } from "@/data/types";

/**
 * Per-agent-kind content for the interactive explainer pages
 * (`components/agents/*` + `pages/AgentDetail.tsx`). Kept as plain data --
 * not JSX -- so the page/components stay one small per-kind lookup instead
 * of a giant conditional. Every claim here is grounded in the actual
 * implementation (`rl/agents/*.py`'s module docstrings and code), not
 * invented: exact hyperparameter defaults (epsilon=1.0/0.05/0.995, the
 * Bellman update), the CSP solver's two real deduction rules, DQN's three
 * documented stabilization tricks, and PPO's actor-critic split are all
 * quoted/paraphrased from the source, never guessed at.
 */

export interface PipelineStepConfig {
  title: string;
  description: string;
  icon: LucideIcon;
  /** A small illustrative example attached to this one step -- always
   * schematic/hand-authored (see `AlgorithmPipeline`'s "illustrative"
   * caption), never presented as a real measured value. */
  example?: { label: string; value: string; highlight?: boolean }[];
}

export interface ArchitectureBranchConfig {
  label: string;
  description: string;
  output: string;
  icon: LucideIcon;
}

export interface ArchitectureConfig {
  input: { label: string; description: string; icon: LucideIcon };
  branches: ArchitectureBranchConfig[];
}

export type DecisionExampleConfig =
  | { mode: "replay"; agentName: string }
  | {
      mode: "illustrative";
      caption: string;
      formula: string;
      rows: { label: string; value: string }[];
    };

export interface AgentExplainerConfig {
  /** One-line "how it decides" summary -- punchier than the catalog description. */
  tagline: string;
  /** Linear step-by-step flow (CSP's deduction pipeline, DQN's forward pass,
   * Q-Learning's update cycle). Omitted for agents better shown as a branching
   * `architecture` instead (PPO). */
  pipeline?: PipelineStepConfig[];
  /** Draws a return arrow from the last step back to the first -- Q-Learning's
   * state/action/reward/update cycle repeats every step. */
  pipelineLoops?: boolean;
  /** Branching network diagram (PPO's actor/critic split). Mutually exclusive
   * with `pipeline` in practice, though nothing enforces that structurally. */
  architecture?: ArchitectureConfig;
  /** Keys into `CONCEPT_GLOSSARY` -- the algorithm-level facts this page explains. */
  concepts: string[];
  decisionExample: DecisionExampleConfig;
}

/**
 * Short, accurate descriptions of algorithm concepts, keyed by label. Some
 * keys exactly match a real recorded technique label from the backend
 * (`results_loader.py`'s `_TECHNIQUE_RULES`, e.g. "Double DQN", "LR decay") --
 * those are both an always-explained concept here AND a dynamic per-run
 * badge in `AgentOverview` when the best run actually recorded it. Others
 * (experience replay, target networks, GAE, epsilon-greedy, ...) are
 * structural facts about this codebase's implementation that are always
 * true, not conditionally flagged by the backend, so they only appear here.
 */
export const CONCEPT_GLOSSARY: Record<string, string> = {
  "Experience replay":
    "Past transitions are stored in a buffer and replayed in random minibatches, breaking the correlation between consecutive steps of the same episode.",
  "Target network":
    "Bootstrapped targets are computed from a periodically-synced copy of the network, not the one being updated every step -- keeps the regression target from shifting under its own gradient.",
  "Double DQN":
    "The online network picks the best next action; a separate target network evaluates it -- decoupling \"which action\" from \"how good\" removes most of vanilla DQN's overestimation bias.",
  "LR decay":
    "An optional schedule that lowers the learning rate over the course of training, for finer-grained convergence later on.",
  "Reduced network capacity":
    "An ablation using a smaller network, to test how much model capacity matters independent of other changes.",
  "Best-checkpoint deployment":
    "Evaluation uses whichever training checkpoint scored the best win rate, not just the final snapshot -- guards against a late unstable patch overwriting a good policy.",
  "Clipped-surrogate PPO":
    "Clips the probability ratio between the new and old policy to a small range, capping how much a single update can exploit one advantage estimate.",
  "Reward shaping":
    "Adds auxiliary reward signals beyond win/lose, giving the agent denser feedback during training.",
  "Generalized Advantage Estimation (GAE)":
    "Blends a low-variance, high-bias one-step estimate with a high-variance, low-bias multi-step estimate of \"how much better than expected was this action,\" trading one off against the other.",
  "Entropy bonus":
    "Adds a small bonus for keeping the action distribution spread out, discouraging the policy from collapsing to one action too early.",
  "Epsilon-greedy exploration":
    "Acts randomly with probability epsilon (starting near 1.0, decaying toward a small floor), otherwise picks the highest-value known action -- balancing exploration against exploiting what's already learned.",
  "Tabular Q-table":
    "Every distinct board pattern is its own lookup key in a plain dictionary -- nothing generalizes to a pattern the agent hasn't seen before, unlike a network's learned features.",
  "Subset deduction rule":
    "If one constraint's cells are a subset of another's, the leftover cells must contain exactly the difference in mine counts -- often enough to resolve them too.",
  "Probability fallback":
    "When no cell can be proven safe, the solver reveals whichever hidden cell has the lowest estimated mine probability instead of guessing blindly.",
  "Random baseline":
    "Picks uniformly among the hidden cells still on the board -- no state, no learning, no memory between moves. The floor every other agent is measured against.",
};

export const AGENT_EXPLAINERS: Record<AgentKind, AgentExplainerConfig> = {
  "rule-based": {
    tagline: "Deduces safe cells and mines from logical constraints -- no learning, no guessing while a certain answer exists.",
    pipeline: [
      {
        title: "Board Observation",
        description: "Reads the current revealed numbers and hidden cells -- nothing else.",
        icon: Eye,
      },
      {
        title: "Generate Constraints",
        description: "Each revealed number becomes a constraint: \"exactly N of these hidden neighbors are mines.\"",
        icon: ListChecks,
      },
      {
        title: "Apply Deduction Rules",
        description:
          "Repeats two rules to a fixpoint: a constraint of 0 makes its cells safe; a constraint equal to its cell count makes them mines; and one constraint's cells being a subset of another's can resolve the difference too.",
        icon: GitMerge,
      },
      {
        title: "Select Safe Cell",
        description: "Reveals a cell it has proven safe. With no certain move left, it falls back to the lowest estimated mine probability.",
        icon: Target,
      },
    ],
    concepts: ["Subset deduction rule", "Probability fallback"],
    decisionExample: { mode: "replay", agentName: "CSP" },
  },

  "q-learning": {
    tagline: "Learns a Q-value for every (state, action) pair directly from trial and error -- no neural network, just a lookup table that grows as it explores.",
    pipeline: [
      {
        title: "State",
        description: "The visible board (hidden/revealed cells) is used directly as a lookup key -- no encoding network.",
        icon: Grid3x3,
      },
      {
        title: "Action",
        description: "Picks a hidden cell with epsilon-greedy selection: usually the highest known Q-value, occasionally random to keep exploring.",
        icon: Target,
      },
      {
        title: "Reward",
        description: "The environment returns a reward for the move, and the resulting next state.",
        icon: Coins,
      },
      {
        title: "Q-table Update",
        description: "Nudges the table toward better long-term estimates using the Bellman update below.",
        icon: RefreshCw,
      },
    ],
    pipelineLoops: true,
    concepts: ["Epsilon-greedy exploration", "Tabular Q-table"],
    decisionExample: {
      mode: "illustrative",
      caption: "One application of the real update rule above -- not a value pulled from a specific recorded run.",
      formula: "Q(s,a) ← Q(s,a) + α · (r + γ · max Q(s′,a′) − Q(s,a))",
      rows: [
        { label: "Before", value: "Q(s,a) = 0.20" },
        { label: "Reward received", value: "+1" },
        { label: "After update", value: "Q(s,a) = 0.35" },
      ],
    },
  },

  dqn: {
    tagline: "Replaces Q-learning's table with a CNN that estimates every cell's Q-value at once, generalizing to board patterns it's never exactly seen.",
    pipeline: [
      {
        title: "11-channel board encoding",
        description: "Each cell's hidden/flag/0-8 state is one-hot encoded across 11 channels, instead of a single raw number.",
        icon: Layers,
      },
      {
        title: "CNN Feature Extractor",
        description: "Convolutional layers scan local neighborhoods of cells, the same way a person reads clue numbers in context.",
        icon: Cpu,
      },
      {
        title: "Q-value output per cell",
        description: "One Q-value per cell in a single forward pass; the agent (when not exploring) picks the hidden cell with the highest value.",
        icon: Target,
        example: [
          { label: "(0,0)", value: "0.12" },
          { label: "(0,1)", value: "0.87 ⭐", highlight: true },
          { label: "(0,2)", value: "0.21" },
        ],
      },
    ],
    concepts: ["Experience replay", "Target network", "Double DQN", "LR decay"],
    decisionExample: { mode: "replay", agentName: "DQN" },
  },

  ppo: {
    tagline: "Learns a policy directly -- \"what should I do here\" -- instead of Q-values, using a second network to judge how good each state is.",
    architecture: {
      input: {
        label: "Observation",
        description: "11-channel board encoding, shared with DQN's CNN input.",
        icon: Layers,
      },
      branches: [
        {
          label: "Actor Network",
          description: "Outputs one logit per cell, masked to hidden cells only, then turned into a probability distribution.",
          output: "Action probabilities",
          icon: Target,
        },
        {
          label: "Critic Network",
          description: "Outputs a single scalar estimate of how good the current state is.",
          output: "State value V(s)",
          icon: Scale,
        },
      ],
    },
    concepts: ["Generalized Advantage Estimation (GAE)", "Clipped-surrogate PPO", "Entropy bonus"],
    decisionExample: { mode: "replay", agentName: "PPO" },
  },

  random: {
    tagline: "No state, no learning -- picks uniformly among the hidden cells still on the board. The floor every other agent is measured against.",
    pipeline: [
      {
        title: "Board Observation",
        description: "Reads which cells are still hidden -- ignores everything else about the board.",
        icon: Eye,
      },
      {
        title: "Select Random Cell",
        description: "Picks uniformly among the hidden cells, with no memory of past moves or outcomes.",
        icon: Target,
      },
    ],
    concepts: ["Random baseline"],
    decisionExample: { mode: "replay", agentName: "Random" },
  },
};
