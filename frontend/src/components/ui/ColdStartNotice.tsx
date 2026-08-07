import { Coffee } from "lucide-react";

/** Shown under a loading skeleton once a request has been pending long enough
 * (see `useApiQuery`'s `isSlow`) that it's more likely a Render free-tier
 * cold start than a normal brief loading flicker -- without this, a first-time
 * visitor hitting the API right after it's spun down just sees empty
 * skeletons for up to a minute with no explanation. */
export function ColdStartNotice() {
  return (
    <div className="flex items-center gap-2 text-sm text-text-muted">
      <Coffee className="h-4 w-4 shrink-0 animate-pulse" aria-hidden="true" />
      <span>Waking up the backend — first load after inactivity can take up to a minute.</span>
    </div>
  );
}
