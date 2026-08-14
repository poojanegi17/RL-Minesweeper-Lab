import { LandingLogoMark } from "@/components/landing/LandingPrimitives";

/** Dark-styled restatement of the site-wide `Footer.tsx`'s real copy --
 * the landing page renders its own chrome (see `Layout.tsx`) rather than
 * the shared light/dark-aware Footer. No nav links here by design -- the
 * top `LandingNavbar` (and the top nav is site-wide now, on every route)
 * already covers navigation, so this stays identity + status only. */
export function LandingFooter() {
  return (
    <footer className="relative z-10 border-t border-white/10">
      <div className="mx-auto flex w-full max-w-6xl items-center gap-2.5 px-6 py-10">
        <LandingLogoMark />
        <div>
          <p className="text-sm font-medium text-white">RL Minesweeper Lab</p>
          <p className="text-xs text-white/40">
            Comparing rule-based and reinforcement learning approaches to Minesweeper.
          </p>
        </div>
      </div>
      <div className="mx-auto w-full max-w-6xl px-6 pb-8 text-xs text-white/30">
        Live results, powered by a FastAPI backend reading directly from the RL training pipeline's own artifacts.
      </div>
    </footer>
  );
}
