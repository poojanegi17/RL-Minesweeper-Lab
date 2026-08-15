import type { HTMLAttributes } from "react";
import { cn } from "@/lib/cn";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  interactive?: boolean;
}

export function Card({ interactive, className, ...props }: CardProps) {
  return (
    <div
      className={cn(
        // `glass-card` (styles/landing.css) owns the surface -- the same glass
        // the home page's research-pipeline card uses. `border` is kept as a
        // width-only utility so the call sites that pass their own
        // `border-primary/20` / `border-white/10` still get a visible edge.
        "glass-card rounded-xl border border-border p-5",
        interactive && "glass-card-interactive transition-all duration-200 hover:-translate-y-0.5",
        className,
      )}
      {...props}
    />
  );
}
