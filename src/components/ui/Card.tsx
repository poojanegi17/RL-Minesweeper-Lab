import type { HTMLAttributes } from "react";
import { cn } from "@/lib/cn";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  interactive?: boolean;
}

export function Card({ interactive, className, ...props }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-xl border border-border bg-surface p-5 shadow-sm shadow-black/[0.02]",
        interactive &&
          "transition-all duration-200 hover:-translate-y-0.5 hover:border-text-muted/50 hover:shadow-md hover:shadow-black/[0.04]",
        className,
      )}
      {...props}
    />
  );
}
