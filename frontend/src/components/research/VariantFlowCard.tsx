import { motion } from "framer-motion";
import { ChevronDown, Trophy } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { VariantStoryCard } from "@/components/research/VariantStoryCard";
import { formatPercent, humanizeVariant, type VariantStory } from "@/lib/experimentAdapters";
import { type Narrative } from "@/components/research/NarrativeText";
import { cn } from "@/lib/cn";

interface VariantFlowCardProps {
  story: VariantStory;
  /** One-line "what this variant is/why it was tried" -- from
   * `VARIANT_NARRATIVES`, or a mechanical fallback when this variant has no
   * hand-authored entry. Shown both as this card's collapsed teaser and,
   * verbatim, as `VariantStoryCard`'s expanded "Why was this tested?" body. */
  about: string;
  /** The real finding for this variant, passed through to `VariantStoryCard`'s
   * "What did we learn?" section -- see that prop's own doc for why. */
  learned?: Narrative;
  isBest: boolean;
  isOpen: boolean;
  onClick: () => void;
  accentColor: string;
  index: number;
}

/**
 * One variant in a DQN/PPO family's story, told the same way
 * `ResearchPipeline`'s top-level agent cards are: collapsed to a name + a
 * one-line "about" teaser, click to expand its full lab-notebook entry
 * (`VariantStoryCard`, header suppressed since this card's own header
 * already shows the name/badges) directly beneath it.
 */
export function VariantFlowCard({ story, about, learned, isBest, isOpen, onClick, accentColor, index }: VariantFlowCardProps) {
  const { runBrief, isBaseline } = story;
  const title = runBrief.variant ? humanizeVariant(runBrief.variant) : runBrief.title;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.3 }}
      transition={{ duration: 0.35, ease: "easeOut", delay: index * 0.05 }}
      className="flex flex-col gap-3"
    >
      <button
        type="button"
        onClick={onClick}
        aria-expanded={isOpen}
        className="block w-full rounded-xl text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60"
      >
        <Card
          interactive
          className={cn("flex w-full items-center gap-4 transition-colors", isOpen && "border-primary/50 shadow-md shadow-primary/10")}
          style={isOpen ? { borderColor: `${accentColor}80` } : undefined}
        >
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-1.5">
              <h4 className="font-semibold text-heading">{title}</h4>
              {isBaseline && <Badge variant="outline">Baseline</Badge>}
              {isBest && (
                <Badge variant="neutral" className="gap-1 text-primary">
                  <Trophy className="h-3 w-3" />
                  Best
                </Badge>
              )}
            </div>
            <p className="mt-0.5 text-sm text-text-muted">{about}</p>
          </div>

          <div className="flex shrink-0 items-center gap-3">
            <span className="hidden font-mono text-sm font-semibold tabular-nums sm:inline" style={{ color: accentColor }}>
              {formatPercent(runBrief.win_rate)}
            </span>
            <ChevronDown className={cn("h-4 w-4 text-text-muted transition-transform", isOpen && "rotate-180 text-primary")} aria-hidden="true" />
          </div>
        </Card>
      </button>

      {isOpen && <VariantStoryCard story={story} about={about} learned={learned} isBest={isBest} accentColor={accentColor} hideHeader />}
    </motion.div>
  );
}
