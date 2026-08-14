import { NoiseFilters } from "@/components/landing/NoiseFilters";
import { LandingHero } from "@/components/landing/LandingHero";
import { WhoLeadsSection } from "@/components/landing/WhoLeadsSection";
import { WatchThemCompete } from "@/components/landing/WatchThemCompete";
import { BeatTheAgents } from "@/components/landing/BeatTheAgents";
import { FindingsGrid } from "@/components/landing/FindingsGrid";
import { MeetTheAgents } from "@/components/landing/MeetTheAgents";
import { ResearchPipelineSection } from "@/components/landing/ResearchPipelineSection";

/**
 * The marketing landing page. `Layout.tsx` now renders `LandingNavbar` and
 * `LandingFooter` site-wide (every route gets the same glassy sticky nav and
 * footer, not just this one) -- this page only owns what's unique to it:
 * the hero and everything between the nav and the footer. Every stat on
 * this page is read live from the same `/api/*` endpoints the rest of the
 * app uses.
 *
 * `PlayableMinesweeper`/`ObservationVisualizer`/`LeaderboardShowcase`/
 * `ResearchJourney` (the previous homepage's other interactive sections) are
 * no longer wired in here -- this page follows a different, marketing-page
 * brief. Their component files are untouched in case they get a new home
 * (`ResearchJourney` was briefly retrieved into `ResearchPipelineSection`,
 * then dropped in favor of a single CTA card -- see that file's comment).
 * `AgentMindsComparison`/`AgentShowcase` themselves *are* still live, just
 * retrieved into `WatchThemCompete`/`MeetTheAgents` below instead of their
 * old spots. `MenuBarStrip`/`ReplayViewerShowcase`/`DecisionBreakdown`/
 * `TechniquesCloud`/`FindingsGrid`/`BoardConfigurations`/`LandingFinalCTA`
 * (components/sections this page used before, some replaced and some
 * removed outright) are likewise unused now, not deleted.
 */
export function Home() {
  return (
    <div className="landing-page relative min-h-screen">
      <NoiseFilters />

      <LandingHero />
      <WhoLeadsSection />
      <WatchThemCompete />
      {/* Play -> meet the agents -> what we found. The 11-channel encoding
       * panel deliberately lives on About rather than here, so the landing
       * page stays a demonstration and the explanation has one home. */}
      <BeatTheAgents />
      <MeetTheAgents />
      <FindingsGrid />
      <ResearchPipelineSection />
    </div>
  );
}
