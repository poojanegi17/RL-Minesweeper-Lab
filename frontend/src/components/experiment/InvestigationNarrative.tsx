import { GitCompare, HelpCircle, Lightbulb } from "lucide-react";
import { Card } from "@/components/ui/Card";
import type { InvestigationNarrative as InvestigationNarrativeData } from "@/lib/experimentAdapters";

interface InvestigationNarrativeProps {
  narrative: InvestigationNarrativeData;
}

const QUESTIONS = [
  { key: "whyTested", question: "Why was this tested?", icon: HelpCircle },
  { key: "whatChanged", question: "What changed?", icon: GitCompare },
  { key: "whatLearned", question: "What did we learn?", icon: Lightbulb },
] as const;

/**
 * Reframes an experiment family as a research investigation -- three
 * questions instead of a flat run listing. Every answer comes from
 * `buildInvestigationNarrative` (`@/lib/experimentAdapters`), which
 * synthesizes real API fields (description, variant labels, techniques,
 * metrics_summary) into these sentences; nothing here is authored per
 * experiment.
 */
export function InvestigationNarrative({ narrative }: InvestigationNarrativeProps) {
  return (
    <div className="grid gap-4 sm:grid-cols-3">
      {QUESTIONS.map(({ key, question, icon: Icon }) => (
        <Card key={key} className="flex flex-col gap-2">
          <div className="flex items-center gap-2 text-text-muted">
            <Icon className="h-4 w-4" />
            <h3 className="text-xs font-medium tracking-wide uppercase">{question}</h3>
          </div>
          <p className="text-sm text-text">{narrative[key]}</p>
        </Card>
      ))}
    </div>
  );
}
