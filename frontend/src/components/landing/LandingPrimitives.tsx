import { type ReactNode } from "react";
import { Link } from "react-router-dom";
import { Bomb, ChevronRight } from "lucide-react";
import { cn } from "@/lib/cn";

/**
 * The site's real logo mark (same `Bomb`-in-a-tile glyph `Header.tsx` uses
 * everywhere else), so the landing page reads as the same product rather
 * than introducing a second, unrelated brand mark for one page.
 */
export function LandingLogoMark({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        "flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-white text-black",
        className,
      )}
    >
      <Bomb className="h-4 w-4" />
    </span>
  );
}

interface GlassPillButtonProps {
  label: string;
  to?: string;
  onClick?: () => void;
  full?: boolean;
  icon?: ReactNode;
  className?: string;
}

/** Rounded-full white pill CTA -- the design brief's "AppleButton" without
 * the Apple trademark, since there's no native app here to badge one with. */
export function GlassPillButton({ label, to, onClick, full, icon, className }: GlassPillButtonProps) {
  const classes = cn(
    "group inline-flex items-center justify-center gap-2 rounded-full bg-white text-black font-medium text-sm px-5 py-3 transition-all hover:bg-white/90 active:scale-[0.98]",
    full && "w-full",
    className,
  );
  const content = (
    <>
      {icon}
      {label}
      <ChevronRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-[1px]" />
    </>
  );

  if (to) {
    return (
      <Link to={to} className={classes}>
        {content}
      </Link>
    );
  }
  return (
    <button type="button" onClick={onClick} className={classes}>
      {content}
    </button>
  );
}

interface SectionEyebrowProps {
  label: string;
  tag?: string;
}

export function SectionEyebrow({ label, tag }: SectionEyebrowProps) {
  return (
    <div className="flex items-center gap-2">
      <span className="flex items-center gap-2 text-sm font-medium text-white/70">
        <span className="h-1.5 w-1.5 rounded-full bg-white" />
        {label}
      </span>
      {tag && (
        <span className="rounded-full border border-white/10 px-2 py-0.5 text-xs text-white/50">{tag}</span>
      )}
    </div>
  );
}
