import { useState, type ReactNode } from "react";
import { motion } from "framer-motion";
import { Gamepad2, Sparkles, Trophy } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { EmptyState } from "@/components/ui/EmptyState";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { SectionCard } from "@/components/home/SectionCard";
import { AmbientBackground } from "@/components/home/AmbientBackground";
import { InteractiveHeroBoard } from "@/components/home/InteractiveHeroBoard";
import { PlayableMinesweeper, type PlayableMinesweeperSummary } from "@/components/home/PlayableMinesweeper";
import { AIComparisonBoard } from "@/components/home/AIComparisonBoard";
import { ObservationVisualizer } from "@/components/home/ObservationVisualizer";
import { AgentMindsComparison } from "@/components/home/AgentMindsComparison";
import { AgentShowcase } from "@/components/home/AgentShowcase";
import { LeaderboardShowcase } from "@/components/home/LeaderboardShowcase";
import { ResearchJourney } from "@/components/home/ResearchJourney";
import { getLeaderboard } from "@/api/metrics";
import { getReplay, getReplays } from "@/api/replays";
import { useApiQuery, type QueryStatus } from "@/hooks/useApiQuery";
import type { LeaderboardEntry } from "@/types/metrics";
import type { ReplayDetail, ReplayStep } from "@/types/replay";

const PLAY_SECTION_ID = "play-yourself";
const WATCH_SECTION_ID = "watch-ai-play";

/** Replays are agent-tagged but not "which is most interesting" -- DQN/PPO's
 * numeric reasoning makes the punchiest preview, CSP's textual inference
 * next best, Random last since it never records a reason at all. */
const REPLAY_AGENT_PREFERENCE = ["DQN", "PPO", "CSP", "Random"];

interface FeaturedReplay {
  replay: ReplayDetail;
  step: ReplayStep;
}

interface HomeData {
  leaderboard: LeaderboardEntry[];
  featuredReplay: FeaturedReplay | null;
}

async function fetchFeaturedReplay(): Promise<FeaturedReplay | null> {
  const replays = await getReplays();
  if (replays.length === 0) return null;

  const chosen =
    REPLAY_AGENT_PREFERENCE.map((agent) => replays.find((r) => r.agent === agent)).find((r) => r !== undefined) ??
    replays[0];
  const replay = await getReplay(chosen.id);
  const step = replay.timeline.find((s) => s.reasoning !== null) ?? replay.timeline[0];
  if (!step) return null;
  return { replay, step };
}

/**
 * One composed fetch for the whole page, shared by every section below that
 * needs real data (the hero's autoplay episode, the observation toggle, the
 * leaderboard, the research journey teaser) -- fetched once, never
 * re-fetched or hardcoded per section. `PlayableMinesweeper` needs no data
 * (a local game), `AIComparisonBoard`/`AgentMindsComparison` each own their
 * own agent-scoped replay fetch via `useAgentReplay`, and `AgentShowcase` is
 * real routing over a fixed, known agent catalog. The full experiment story
 * per agent lives on `/research` now, fetched there instead.
 */
async function fetchHomeData(): Promise<HomeData> {
  const [leaderboard, featuredReplay] = await Promise.all([getLeaderboard(), fetchFeaturedReplay()]);
  return { leaderboard, featuredReplay };
}

function scrollToId(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

export function Home() {
  const { data, status, error, isSlow, retry } = useApiQuery(fetchHomeData, []);
  const [humanSummary, setHumanSummary] = useState<PlayableMinesweeperSummary>({
    status: "idle",
    revealedCount: 0,
    flagCount: 0,
  });

  return (
    <div className="flex flex-col gap-10">
      <AmbientBackground />

      <HeroSection featuredReplay={data?.featuredReplay ?? null} status={status} isSlow={isSlow} />

      <SectionCard title="How does AI see Minesweeper?" glow="bg-blue-500/10">
        <ObservationVisualizer board={data?.featuredReplay?.step.board_state ?? null} />
      </SectionCard>

      <SectionCard
        id={PLAY_SECTION_ID}
        title="Play Minesweeper yourself"
        description="You solve Minesweeper using intuition. Agents solve it using observations and decisions."
        glow="bg-emerald-500/10"
      >
        <PlayableMinesweeper onStateChange={setHumanSummary} />
      </SectionCard>

      <SectionCard id={WATCH_SECTION_ID} title="Now watch AI play" glow="bg-agent-dqn/10">
        <AIComparisonBoard humanSummary={humanSummary} />
      </SectionCard>

      <SectionCard title="Different minds, same game" glow="bg-agent-ppo/10">
        <AgentMindsComparison />
      </SectionCard>

      <SectionCard title="Meet the agents" glow="bg-agent-rule-based/10">
        <AgentShowcase />
      </SectionCard>

      <SectionCard title="Live leaderboard" glow="bg-amber-400/10">
        <DataSection
          status={status}
          error={error}
          isSlow={isSlow}
          retry={retry}
          loadingFallback={<LeaderboardSkeleton />}
          errorTitle="Couldn't load the leaderboard"
        >
          {data && data.leaderboard.length === 0 ? (
            <EmptyState icon={Trophy} title="No leaderboard data" description="The backend didn't return any ranked agents." />
          ) : (
            data && <LeaderboardShowcase leaderboard={data.leaderboard} />
          )}
        </DataSection>
      </SectionCard>

      <SectionCard
        title="Research journey"
        description="Five algorithms, one evolving pipeline — click any milestone to open its full experiment story."
        glow="bg-agent-q-learning/10"
      >
        <DataSection
          status={status}
          error={error}
          isSlow={isSlow}
          retry={retry}
          loadingFallback={<Skeleton className="h-56 w-full" />}
          errorTitle="Couldn't load the leaderboard"
        >
          {data && <ResearchJourney leaderboard={data.leaderboard} />}
        </DataSection>
      </SectionCard>
    </div>
  );
}

/** Shared loading/error/success gate for every section fed by `fetchHomeData` --
 * keeps that boilerplate out of `Home()`'s own body. */
function DataSection({
  status,
  error,
  isSlow,
  retry,
  loadingFallback,
  errorTitle,
  children,
}: {
  status: QueryStatus;
  error: Error | null;
  isSlow: boolean;
  retry: () => void;
  loadingFallback: ReactNode;
  errorTitle: string;
  children: ReactNode;
}) {
  if (status === "loading") {
    return (
      <div className="flex flex-col gap-3">
        {loadingFallback}
        {isSlow && <ColdStartNotice />}
      </div>
    );
  }
  if (status === "error" && error) return <ApiErrorState error={error} onRetry={retry} title={errorTitle} />;
  if (status === "success") return <>{children}</>;
  return null;
}

function LeaderboardSkeleton() {
  return (
    <div className="flex flex-col gap-3" aria-label="Loading leaderboard">
      {Array.from({ length: 5 }).map((_, index) => (
        <Skeleton key={index} className="h-16 w-full" />
      ))}
    </div>
  );
}

// --- Hero --------------------------------------------------------------------

function HeroSection({
  featuredReplay,
  status,
  isSlow,
}: {
  featuredReplay: FeaturedReplay | null;
  status: QueryStatus;
  isSlow: boolean;
}) {
  return (
    <SectionCard>
      <div className="grid items-center gap-12 lg:grid-cols-2">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className="flex flex-col items-start gap-6"
        >
          <span className="inline-flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1 font-mono text-[11px] tracking-wide text-text-muted uppercase shadow-sm shadow-primary/10">
            <span className="h-1.5 w-1.5 rounded-full bg-primary shadow-[0_0_8px_2px] shadow-primary/60" />
            Interactive AI laboratory
          </span>

          <h1 className="text-4xl leading-[1.08] font-semibold tracking-tight text-heading sm:text-5xl lg:text-[3.25rem]">
            Can You Teach an AI to Play Minesweeper?
          </h1>
          <p className="max-w-md text-lg leading-relaxed text-text-muted">
            Some agents reason with logic. Others learn from experience. Play the board yourself, then watch how
            each one thinks.
          </p>
          <div className="flex flex-wrap gap-3">
            <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}>
              <Button onClick={() => scrollToId(PLAY_SECTION_ID)}>
                <Gamepad2 className="h-4 w-4" />
                Play Minesweeper Yourself
              </Button>
            </motion.div>
            <motion.div whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}>
              <Button variant="secondary" onClick={() => scrollToId(WATCH_SECTION_ID)}>
                <Sparkles className="h-4 w-4" />
                Watch AI Solve
              </Button>
            </motion.div>
          </div>
        </motion.div>

        <div>
          {status === "loading" ? (
            <div className="flex flex-col items-center gap-3">
              <Skeleton className="h-72 w-72" />
              {isSlow && <ColdStartNotice />}
            </div>
          ) : (
            <InteractiveHeroBoard replay={featuredReplay?.replay ?? null} />
          )}
        </div>
      </div>
    </SectionCard>
  );
}
