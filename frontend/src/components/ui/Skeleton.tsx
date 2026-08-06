import type { HTMLAttributes } from "react";
import { cn } from "@/lib/cn";

/** A pulsing placeholder block for loading states, styled with the same
 * surface tokens as `Card` so skeletons sit naturally alongside real content. */
export function Skeleton({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("animate-pulse rounded-lg bg-surface-hover", className)}
      {...props}
    />
  );
}
