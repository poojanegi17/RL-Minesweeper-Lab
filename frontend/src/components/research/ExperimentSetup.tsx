import { useRef, useState } from "react";
import { FlaskConical } from "lucide-react";
import { AlgorithmPipeline } from "@/components/agents/AlgorithmPipeline";
import { VariantStoryCard } from "@/components/research/VariantStoryCard";
import { VariantTimeline } from "@/components/research/VariantTimeline";
import { buildVariantStories } from "@/lib/experimentAdapters";
import { AGENT_EXPLAINERS } from "@/lib/agentExplainers";
import { AGENT_STYLES, type AgentKind } from "@/data/types";
import type { ExperimentDetail, ExperimentSummary } from "@/types/experiment";

interface ExperimentSetupProps {
  agentName: string;
  kind: AgentKind;
  family: ExperimentSummary | null;
  variantDetails: ExperimentDetail[];
  accentColor: string;
}

/**
 * "How did we experiment with it?" -- for DQN/PPO, a clickable evolution
 * timeline (`VariantTimeline`) above one lab-notebook card per real training
 * variant (`buildVariantStories`); selecting a timeline pill scrolls to and
 * highlights the matching card. For Random/CSP/Q-Learning, which write no
 * training ablation at all, an honest note plus the agent's real decision
 * pipeline instead of a fabricated experiment list.
 */
export function ExperimentSetup({ agentName, kind, family, variantDetails, accentColor }: ExperimentSetupProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const cardRefs = useRef(new Map<string, HTMLDivElement>());

  if (family) {
    const stories = buildVariantStories(family.runs, variantDetails);
    const bestRunId = family.metrics_summary.best_run_id;

    function handleSelect(runId: string) {
      setSelectedId(runId);
      cardRefs.current.get(runId)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    return (
      <div className="flex flex-col gap-5">
        <VariantTimeline stories={stories} selectedId={selectedId} onSelect={handleSelect} accentColor={accentColor} />

        <div className="grid gap-4 sm:grid-cols-2">
          {stories.map((story) => (
            <div key={story.runBrief.id} ref={(el) => void (el ? cardRefs.current.set(story.runBrief.id, el) : cardRefs.current.delete(story.runBrief.id))}>
              <VariantStoryCard
                story={story}
                baselineDescription={family.description}
                isBest={story.runBrief.id === bestRunId}
                isSelected={story.runBrief.id === selectedId}
                accentColor={accentColor}
              />
            </div>
          ))}
        </div>
      </div>
    );
  }

  const style = AGENT_STYLES[kind];
  const explainer = AGENT_EXPLAINERS[kind];

  return (
    <div className="flex flex-col gap-4">
      <p className="flex items-center gap-2 text-sm text-text-muted">
        <FlaskConical className="h-4 w-4 shrink-0" />
        No formal training ablation exists for {agentName} — here's how it decides instead.
      </p>
      {explainer.pipeline && (
        <AlgorithmPipeline steps={explainer.pipeline} loops={explainer.pipelineLoops} accentClassName={style.text} />
      )}
    </div>
  );
}
