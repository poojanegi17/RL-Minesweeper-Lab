import { FlaskConical, Lightbulb, ListChecks } from "lucide-react";
import { AlgorithmPipeline } from "@/components/agents/AlgorithmPipeline";
import { ArchitectureDiagram } from "@/components/agents/ArchitectureDiagram";
import { LevelPipelineLink } from "@/components/research/LevelPipelineLink";
import { BoardConfigComparisonTable } from "@/components/research/BoardConfigComparisonTable";
import { AGENT_EXPLAINERS } from "@/lib/agentExplainers";
import { ARCHITECTURE_RATIONALE } from "@/lib/variantNarratives";
import { LEVEL_PIPELINE_IDS, type PipelineLevel } from "@/lib/levelPipelines";
import { AGENT_STYLES, type AgentKind } from "@/data/types";

interface ExperimentSetupProps {
  agentName: string;
  kind: AgentKind;
  accentColor: string;
}

/** A bullet, optionally led by a short bolded term it defines -- used where a
 * point introduces a named setting ("First click safe — ...") rather than
 * simply stating a fact, so the name is scannable without the reader parsing
 * the sentence first. */
type MethodologyPoint = string | { label: string; text: string };

interface MethodologyDetail {
  /** One-line summary shown above the bullet list. */
  intro: string;
  points: MethodologyPoint[];
}

const LEVEL_ORDER: PipelineLevel[] = ["beginner", "intermediate", "expert"];

/**
 * The real training/evaluation procedure behind this agent's win rate --
 * grounded in `agents/q_learning_agent.py`'s actual constructor defaults
 * (alpha=0.1, gamma=0.9, epsilon 1.0 decaying by 0.995/episode to a 0.05
 * floor -- reached after ~598 episodes, computed directly rather than
 * estimated) and `evaluate_agents.py`/`evaluate_board_config.py`'s shared,
 * uniform constants (2,000 greedy evaluation episodes and seed 42 everywhere;
 * Q-Learning's 100,000 training episodes, raised from 20,000 after a budget sweep
 * showed Beginner/Standard still climbing well past the old figure; it persists no
 * checkpoint and is retrained from scratch every run, the same fact its own
 * Research Decision limitation already states). Only Random/CSP/Q-Learning ever reach
 * the no-pipeline branch below, so this key set is exhaustive -- DQN/PPO's
 * training/eval counts already show in their own variant cards instead.
 *
 * Every figure here is a 2,000-episode result. Keep that in step with
 * `evaluate_board_config.EVAL_EPISODES` -- at these win rates a few hundred
 * episodes is a handful of wins, wide enough to reorder cells (see the
 * README's "Evaluation sample size").
 */
const EVALUATION_METHODOLOGY: Partial<Record<AgentKind, MethodologyDetail>> = {
  random: {
    intro:
      "Random has nothing to train -- every game uses the same fixed policy. What matters here is how it is run, and what the two opening-click settings underneath every number on this page actually mean.",
    points: [
      "No parameters, no training episodes. Every hidden cell is equally likely to be picked, every move. It never reads the revealed numbers -- the only thing it uses the board for is to avoid re-clicking a cell that is already open.",
      "Evaluated at every board size and density: 2,000 episodes, fixed seed 42, on a freshly seeded environment. Every agent faces the identical 2,000 boards at a given cell, so a difference between agents is a difference in play rather than luck of the draw.",
      {
        label: "First click safe",
        text: "The setting the current results use. The opening click always opens a mine-free 3x3 block, so no game can be lost on move one and every game starts with a free cascade already on the board.",
      },
      {
        label: "First click unsafe",
        text: "Mines are placed before the first click, so the opening move can hit one. On the 5x5 benchmark board that alone decides about a fifth of all games before any agent has made a real decision.",
      },
      "These are two different games, not two difficulty settings. A win rate measured under one cannot be compared to a win rate measured under the other, which is why the table below switches between them instead of showing both at once.",
      "Random is the clearest illustration of why that matters. The same policy, unchanged, scores 0.45% with an unsafe opening and 1.30% with a safe one -- roughly triple, bought entirely by the environment.",
    ],
  },
  "rule-based": {
    intro:
      "CSP has nothing to train -- its two deduction rules are fixed logic, not learned weights. What matters here is how a move is chosen, and why the opening-click setting turns out to matter more to this agent than any rule in its solver.",
    points: [
      "No parameters, no training episodes. The same two rules (see the flow above) run identically on every board and are re-derived from scratch every move -- nothing carries over between games.",
      "When a cell can be proven safe, it is revealed. That part is exact: if CSP says safe, it is safe.",
      "When nothing can be proven, it falls back to the hidden cell with the lowest estimated mine probability. That estimate is rough -- it averages each touching constraint's count over its size instead of working out which mine layouts are actually possible -- and ties are broken by a seeded random pick.",
      "Evaluated at every board size and density: 2,000 episodes, fixed seed 42, the same boards every other agent faces.",
      {
        label: "First click unsafe",
        text: "Mines are placed before the opening click. On an all-hidden board there are no revealed numbers to reason from, so CSP's first move is a pure coin flip -- literally Random's move -- and 20.65% of games ended there, before its solver ever ran.",
      },
      {
        label: "First click safe",
        text: "The opening click always opens a mine-free 3x3 block. No game is lost on move one, and the free cascade hands the solver a block of revealed numbers to start deducing from.",
      },
      "This setting is worth more to CSP than any rule it has: 43.40% to 70.35% on the benchmark board, with the agent completely unchanged.",
      "And it helps twice over. The cascade seeds the constraint graph, so CSP is forced into fewer gambles (3.16 per game down to 2.27) and the gambles it does face are safer (17.91% fatal down to 13.05%). It is not merely surviving the first move -- it is being handed a better-structured problem.",
    ],
  },
  "q-learning": {
    intro:
      "Q-Learning is the first agent here that actually trains -- \"no formal training ablation\" above means no variant comparison exists, not that it skipped training. What matters is how it stores what it learns, since that single choice decides both its training budget and which board sizes it can be run on at all.",
    points: [
      "The state is the visible board itself, flattened into a tuple and used directly as a dictionary key. No encoding, no features, no notion of similarity -- 4 mines here and the same 4 mines shifted one cell over are unrelated entries.",
      "Trained from scratch at the exact board size and density being evaluated. Nothing is reused: a table keyed by exact board pattern transfers to no other density, and no table is ever saved.",
      "Updated after every step with the Bellman rule — Q(s,a) ← Q(s,a) + α · (r + γ · max Q(s′,a′) − Q(s,a)) — using α (learning rate) = 0.1 and γ (discount factor) = 0.9. The next-state max covers only cells still hidden, and is dropped on a terminal step since there is no future return to bootstrap from.",
      "Exploration is epsilon-greedy: starts fully random, multiplied by 0.995 each episode, floored at 0.05 -- which it reaches by episode 598. Over 99% of training therefore runs near-greedy, so extra episodes buy state coverage rather than exploration.",
      {
        label: "First click unsafe",
        text: "Mines are placed before the opening click, so the first move can lose outright. Scores 1.90% on the benchmark board.",
      },
      {
        label: "First click safe",
        text: "The opening click always opens a mine-free 3x3 block, so no game is lost on move one and every game starts from a cascade. Scores 71.70% on the same board.",
      },
      "Both are trained for 100,000 episodes, so that comparison is like-for-like -- the only thing differing between those two numbers is the board distribution.",
      "100,000 is the project's standard budget for this agent, chosen from a sweep: sparse saturates by 50,000, standard is still improving at 200,000, and dense peaks at 50,000 and then declines. One budget is kept across all densities so the three stay comparable.",
      "Evaluated the same way in both cases: 2,000 episodes with exploration off (always the highest known value), fixed seed 42, on a freshly seeded environment rather than the one it trained in -- so it faces the identical 2,000 boards as every other agent at that cell.",
      {
        label: "Beginner only, deliberately",
        text: "It is never run at Intermediate or Expert. A 9x9 or 16x16 board has vastly more distinct patterns than 5x5, so any realistic budget would leave nearly every evaluation state unvisited -- and an unvisited state returns all-zero values and picks at random. The run would be a slow, expensive way to reproduce the Random baseline, not a test of anything.",
      },
    ],
  },
};

/**
 * "How did we experiment with it, and where did it land?" -- for DQN/PPO,
 * the architecture diagram first (identical across every level/variant below
 * -- only the training config changes, so it belongs once at the top, not
 * repeated per card), then a "why 25,000 episodes everywhere" methodology
 * note, then one `LevelPipeline` per board size that's actually been
 * explored (Beginner always; Intermediate/Expert once trained -- see
 * `LEVEL_PIPELINE_IDS`). Each level tells its own version of the same story
 * `ResearchPipeline`'s top-level agent cards already tell: real variants as
 * clickable cards, what was still wrong with each leading into the next,
 * closing with that level's best-found configuration -- which becomes the
 * *next* level's own starting "baseline" card. Finally
 * `BoardConfigComparisonTable`, spanning every level/density this agent has
 * any data for. For Random/CSP/Q-Learning, which write no training
 * *ablation* (no variant comparison -- Q-Learning is still actually trained,
 * just not compared against alternate configs): a note saying so, the
 * agent's real decision pipeline, the full training/evaluation procedure and
 * parameters (`EVALUATION_METHODOLOGY`), then the same
 * `BoardConfigComparisonTable` -- instead of a fabricated experiment list.
 */
export function ExperimentSetup({ agentName, kind, accentColor }: ExperimentSetupProps) {
  const style = AGENT_STYLES[kind];
  const explainer = AGENT_EXPLAINERS[kind];
  const pipelineIds = LEVEL_PIPELINE_IDS[agentName];

  if (pipelineIds) {
    const rationale = ARCHITECTURE_RATIONALE[agentName];

    return (
      <div className="flex flex-col gap-5">
        {rationale && (
          <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface-hover/40 px-4 py-3">
            <p className="flex items-center gap-2 text-xs font-medium tracking-wide text-text-muted uppercase">
              <Lightbulb className="h-4 w-4 shrink-0" aria-hidden="true" />
              Why does the baseline already look like this?
            </p>
            <ul className="flex flex-col gap-2">
              {rationale.map((point) => (
                <li key={point} className="flex gap-2.5 text-sm text-text">
                  <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-text-muted" aria-hidden="true" />
                  <span>{point}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {explainer.architecture ? (
          <ArchitectureDiagram architecture={explainer.architecture} accentClassName={style.text} />
        ) : explainer.pipeline ? (
          <AlgorithmPipeline steps={explainer.pipeline} loops={explainer.pipelineLoops} accentClassName={style.text} />
        ) : null}

        {/* The text is wrapped in a single <span> deliberately: this is a flex
            container, so a bare text node and the inline <code> would each
            become their own flex item and `gap-2` would split the sentence into
            two separated blocks. `items-start` keeps the icon on the first line
            rather than vertically centred against a multi-line paragraph. */}
        <p className="flex items-start gap-2 text-sm text-text-muted">
          <FlaskConical className="mt-0.5 h-4 w-4 shrink-0" />
          <span>
            The board encoding and the learning rule stay fixed across every level and variant below. One structural
            change was tested and kept — the <code className="font-mono text-xs">fully_conv</code> head, which swaps
            the board-size-specific Linear layer for 1x1 convolutions so one model runs at any board size.
          </span>
        </p>

        {LEVEL_ORDER.filter((level) => pipelineIds[level]).map((level) => (
          <LevelPipelineLink key={level} level={level} agentName={agentName} accentColor={accentColor} />
        ))}

        <BoardConfigComparisonTable agentName={agentName} kind={kind} accentColor={accentColor} />
      </div>
    );
  }

  const methodology = EVALUATION_METHODOLOGY[kind];

  return (
    <div className="flex flex-col gap-4">
      <p className="flex items-center gap-2 text-sm text-text-muted">
        <FlaskConical className="h-4 w-4 shrink-0" />
        No training-variant ablation exists for {agentName} (unlike DQN/PPO's per-level pipelines) — here's how it
        decides.
      </p>

      {explainer.pipeline && (
        <AlgorithmPipeline steps={explainer.pipeline} loops={explainer.pipelineLoops} accentClassName={style.text} />
      )}

      {methodology && (
        <div className="flex flex-col gap-2 rounded-lg border border-border bg-surface-hover/40 px-4 py-3">
          <p className="flex items-center gap-2 text-xs font-medium tracking-wide text-text-muted uppercase">
            <ListChecks className="h-4 w-4 shrink-0" aria-hidden="true" />
            How it was trained and evaluated
          </p>
          <p className="text-sm text-text">{methodology.intro}</p>
          <ul className="mt-1 flex flex-col gap-2">
            {methodology.points.map((point) => {
              const isLabelled = typeof point !== "string";
              return (
                <li key={isLabelled ? point.label : point} className="flex gap-2.5 text-sm text-text">
                  <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-text-muted" aria-hidden="true" />
                  <span>
                    {isLabelled && <strong className="font-semibold text-heading">{point.label} — </strong>}
                    {isLabelled ? point.text : point}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      <BoardConfigComparisonTable agentName={agentName} kind={kind} accentColor={accentColor} />
    </div>
  );
}
