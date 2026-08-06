import { motion } from "framer-motion";
import { humanizeVariant, type VariantStory } from "@/lib/experimentAdapters";
import { cn } from "@/lib/cn";

interface VariantTimelineProps {
  stories: VariantStory[];
  selectedId: string | null;
  onSelect: (runId: string) => void;
  accentColor: string;
}

/**
 * The "evolution timeline inside each algorithm" -- one clickable pill per
 * real training variant, in the same order the family's runs already come
 * in (baseline first). Selecting a pill just highlights/scrolls to the
 * matching `VariantStoryCard` below; it carries no data of its own beyond
 * `VariantStory`.
 */
export function VariantTimeline({ stories, selectedId, onSelect, accentColor }: VariantTimelineProps) {
  return (
    <div className="flex flex-wrap items-center gap-x-1 gap-y-2 sm:flex-nowrap sm:overflow-x-auto">
      {stories.map((story, index) => {
        const isSelected = story.runBrief.id === selectedId;
        const label = story.runBrief.variant ? humanizeVariant(story.runBrief.variant) : story.runBrief.title;

        return (
          <div key={story.runBrief.id} className="flex shrink-0 items-center gap-1">
            <button
              type="button"
              onClick={() => onSelect(story.runBrief.id)}
              className={cn(
                "rounded-full border px-3 py-1.5 text-xs font-medium whitespace-nowrap transition-colors",
                isSelected ? "border-transparent text-white" : "border-border text-text-muted hover:text-text",
              )}
              style={isSelected ? { backgroundColor: accentColor } : undefined}
            >
              {label}
            </button>
            {index < stories.length - 1 && (
              <motion.span
                className="h-px w-4 shrink-0 bg-border"
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                style={{ transformOrigin: "left" }}
                aria-hidden="true"
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
