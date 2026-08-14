import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { slugifyAgentName } from "@/lib/agentAdapters";
import { LEVEL_LABELS_FULL, type PipelineLevel } from "@/lib/levelPipelines";
import { chaptersFor } from "@/lib/pipelineChapters";

interface LevelPipelineLinkProps {
  level: PipelineLevel;
  agentName: string;
  accentColor: string;
}

/** A one-line summary of what this level's story contains, so the link says
 * what it leads to rather than just naming the level. Chapter counts come
 * from the real chapter list; levels without one describe themselves
 * generically, since the run count isn't known until the page fetches. */
function summarize(agentName: string, level: PipelineLevel): string {
  const chapters = chaptersFor(agentName, level);
  if (chapters) {
    return `${chapters.length} chapters, from the first diverging baseline to where the result stops holding — plus every run in full.`;
  }
  return "Every configuration tried at this board size, what each showed, and what was still wrong with it.";
}

/**
 * Links into one board size's full research story
 * (`/research/:agentSlug/:level`).
 *
 * Replaces the accordion these levels used to open inline. That worked while
 * a level was two or three runs; it stopped working once DQN Beginner became
 * five chapters plus ten runs with their own training curves, all nested two
 * accordions deep inside a chamber. A route also makes a level linkable and
 * shareable, which the nested version never was.
 */
export function LevelPipelineLink({ level, agentName, accentColor }: LevelPipelineLinkProps) {
  return (
    <Link
      to={`/research/${slugifyAgentName(agentName)}/${level}`}
      className="block rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60"
    >
      <Card interactive className="flex w-full items-center justify-between gap-4 transition-colors">
        <div className="min-w-0">
          <h3 className="font-semibold text-heading">{LEVEL_LABELS_FULL[level]}</h3>
          <p className="mt-0.5 text-sm text-text-muted">{summarize(agentName, level)}</p>
        </div>
        <ArrowRight className="h-4 w-4 shrink-0" style={{ color: accentColor }} aria-hidden="true" />
      </Card>
    </Link>
  );
}
