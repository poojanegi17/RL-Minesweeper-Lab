import type { ReactNode } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/cn";

interface SectionCardProps {
  id?: string;
  /** Omitted only by the hero, which supplies its own `<h1>` instead of this
   * shell's small uppercase section label. */
  title?: string;
  description?: string;
  children: ReactNode;
  className?: string;
  /** A Tailwind `bg-color/opacity` class for this section's corner glow -- lets each
   * "exhibit panel" carry its own tint (e.g. the agent-comparison section
   * glowing amber, the leaderboard glowing gold) instead of every section
   * looking like an identical stamped-out card. Defaults to the primary
   * accent, same as before this got a prop. */
  glow?: string;
}

/**
 * The homepage's one section shell -- every major section wraps in this
 * rather than a bare `<section>`, so the whole page reads as a row of
 * distinct "exhibit panels" instead of plain black-on-black text blocks. A
 * translucent, blurred surface (vs. the solid `bg-surface` inner `Card`
 * components already use) keeps depth between "the panel" and "the cards
 * inside it." A thin top gradient line and a soft corner glow (tintable per
 * section via `glow`) give each panel its own quiet identity without
 * changing the shared shell's structure.
 */
export function SectionCard({ id, title, description, children, className, glow = "bg-primary/10" }: SectionCardProps) {
  return (
    <motion.section
      id={id}
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      whileHover={{ y: -2 }}
      viewport={{ once: true, amount: 0.15 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className={cn(
        "relative overflow-hidden rounded-2xl border border-border bg-gradient-to-b from-surface/80 to-surface/50 p-6 shadow-xl shadow-black/[0.08] backdrop-blur-sm transition-[border-color,box-shadow] duration-300 hover:border-text-muted/40 hover:shadow-2xl sm:p-8",
        className,
      )}
    >
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-border to-transparent" aria-hidden="true" />
      <div className={cn("pointer-events-none absolute -top-24 -right-24 -z-10 h-56 w-56 rounded-full blur-3xl", glow)} aria-hidden="true" />

      {title && (
        <div className="mb-6">
          <div className="flex items-center gap-2">
            <span className="h-1 w-1 rounded-full bg-primary" aria-hidden="true" />
            <h2 className="text-sm font-medium tracking-[0.08em] text-text-muted uppercase">{title}</h2>
          </div>
          {description && <p className="mt-2 max-w-2xl text-text-muted">{description}</p>}
        </div>
      )}

      {children}
    </motion.section>
  );
}
