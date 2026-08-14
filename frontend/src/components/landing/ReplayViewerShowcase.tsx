import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, Sparkles, Trophy, XCircle } from "lucide-react";
import { ReplayBoard } from "@/components/replay/ReplayBoard";
import { ReplayControls } from "@/components/replay/ReplayControls";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { Skeleton } from "@/components/ui/Skeleton";
import { ColdStartNotice } from "@/components/ui/ColdStartNotice";
import { AGENT_ICONS } from "@/components/agent/agentIcons";
import { SectionEyebrow } from "@/components/landing/LandingPrimitives";
import { getLeaderboard } from "@/api/metrics";
import { getReplay, getReplays } from "@/api/replays";
import { getBoardConfigs } from "@/api/boardConfigs";
import { useApiQuery } from "@/hooks/useApiQuery";
import { agentKindFromName } from "@/lib/agentAdapters";
import { pickBestReplay } from "@/lib/replaySelection";
import { describeDecisionReason } from "@/lib/reasoning";
import { formatPercent } from "@/lib/experimentAdapters";
import { AGENT_HEX } from "@/data/types";
import type { LeaderboardEntry } from "@/types/metrics";
import type { ReplayDetail, ReplaySummary } from "@/types/replay";
import type { BoardLevelInfo } from "@/types/boardConfig";

// Live-data centerpiece config: this specific (level, density) pair is the
// one place both distributions genuinely agree -- every one of the 5 agents
// has a real `board_result` entry here (see `evaluate_board_config.py`), so
// win rates are directly comparable across agents. The default *unscoped*
// leaderboard mixes v1 (DQN/PPO's training-run artifacts) and v2
// (CSP/Q-Learning/Random's re-baselined figures) on one axis -- see
// `routes/metrics.py`'s "MIXED AXIS" note -- which this sidesteps entirely.
const LEVEL = "beginner";
const DENSITY = "sparse";
const AGENT_ORDER = ["CSP", "Q-Learning", "DQN", "Random", "PPO"];
const EVAL_EPISODES = 2000;

interface ShowcaseData {
  leaderboard: LeaderboardEntry[];
  replays: ReplaySummary[];
  boardConfig: BoardLevelInfo | undefined;
}

/**
 * The landing page's visual centerpiece: a live, real-data replay viewer --
 * not a static mockup. Pick an agent, pick one of its actually-recorded
 * episodes, and step through the exact board states and reasoning it
 * produced. Reuses the same `ReplayBoard`/`ReplayControls` the real Replay
 * Viewer at `/replay` renders with, wrapped in a forced `.dark` scope so it
 * always matches this page's fixed-dark aesthetic regardless of the site's
 * light/dark toggle.
 */
export function ReplayViewerShowcase() {
  const [selectedAgent, setSelectedAgent] = useState("CSP");
  const [selectedReplayId, setSelectedReplayId] = useState<string | null>(null);

  const { data, status, error, isSlow, retry } = useApiQuery<ShowcaseData>(async () => {
    const [leaderboard, replays, boardConfigs] = await Promise.all([
      getLeaderboard(LEVEL, DENSITY),
      getReplays(LEVEL, DENSITY),
      getBoardConfigs(),
    ]);
    return { leaderboard, replays, boardConfig: boardConfigs.find((c) => c.level === LEVEL) };
  }, []);

  const agentReplays = useMemo(
    () => (data ? data.replays.filter((r) => r.agent === selectedAgent) : []),
    [data, selectedAgent],
  );

  // Whenever the selected agent changes (including the first successful
  // load), default to its most representative episode -- a real win over a
  // loss, then the longest one -- rather than whatever sorted first.
  useEffect(() => {
    if (agentReplays.length === 0) {
      setSelectedReplayId(null);
      return;
    }
    setSelectedReplayId((current) => {
      if (current && agentReplays.some((r) => r.id === current)) return current;
      return pickBestReplay(agentReplays)?.id ?? null;
    });
  }, [agentReplays]);

  const { data: replayDetail, status: replayStatus } = useApiQuery<ReplayDetail | null>(async () => {
    if (!selectedReplayId) return null;
    return getReplay(selectedReplayId, LEVEL, DENSITY);
  }, [selectedReplayId]);

  const mines = data?.boardConfig?.densities[DENSITY];
  const contextLine =
    data?.boardConfig && mines != null
      ? `${data.boardConfig.rows}×${data.boardConfig.cols} board · ${mines} mines · ${EVAL_EPISODES.toLocaleString()} evaluation episodes`
      : null;

  return (
    <section id="replay-viewer" className="relative z-10 mx-auto max-w-6xl px-6 py-16 md:py-24">
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        className="liquid-glass relative overflow-hidden rounded-3xl border border-white/10 shadow-[0_0_120px_-20px_rgba(0,210,255,0.25)]"
      >
        <div className="flex items-center gap-2 border-b border-white/10 bg-black/40 px-4 py-3">
          <span className="h-3 w-3 rounded-full bg-[#ff5f57]" />
          <span className="h-3 w-3 rounded-full bg-[#febc2e]" />
          <span className="h-3 w-3 rounded-full bg-[#28c840]" />
          <span className="mx-auto text-xs text-white/50">RL Minesweeper Lab — Replay Viewer</span>
          <span className="rounded-full border border-white/10 px-2 py-0.5 text-[10px] tracking-wide text-white/50 uppercase">
            Live data
          </span>
        </div>

        <div className="border-b border-white/10 px-5 py-4">
          <SectionEyebrow label="Watch how each agent actually decided" tag={contextLine ?? undefined} />
        </div>

        {status === "loading" && (
          <div className="flex flex-col gap-3 p-6">
            <Skeleton className="h-[420px] w-full" />
            {isSlow && <ColdStartNotice />}
          </div>
        )}
        {status === "error" && error && (
          <div className="p-6">
            <ApiErrorState error={error} onRetry={retry} title="Couldn't load the replay viewer" />
          </div>
        )}

        {status === "success" && data && (
          <div className="grid min-h-[520px] grid-cols-1 md:grid-cols-12">
            <AgentSidebar
              leaderboard={data.leaderboard}
              selectedAgent={selectedAgent}
              onSelect={setSelectedAgent}
            />
            <EpisodeList
              replays={agentReplays}
              selectedReplayId={selectedReplayId}
              onSelect={setSelectedReplayId}
            />
            <ReaderPane agent={selectedAgent} detail={replayDetail ?? null} status={replayStatus} />
          </div>
        )}
      </motion.div>

      <div className="mt-4 flex justify-center">
        <Link
          to="/replay"
          className="inline-flex items-center gap-1.5 text-sm font-medium text-white/60 transition-colors hover:text-white"
        >
          Open the full replay viewer
          <ArrowRight className="h-3.5 w-3.5" />
        </Link>
      </div>
    </section>
  );
}

function AgentSidebar({
  leaderboard,
  selectedAgent,
  onSelect,
}: {
  leaderboard: LeaderboardEntry[];
  selectedAgent: string;
  onSelect: (agent: string) => void;
}) {
  return (
    <div className="border-b border-white/10 p-3 md:col-span-3 md:border-r md:border-b-0">
      {AGENT_ORDER.map((name) => {
        const kind = agentKindFromName(name);
        const Icon = AGENT_ICONS[kind];
        const entry = leaderboard.find((e) => e.agent === name);
        const isActive = name === selectedAgent;
        return (
          <button
            key={name}
            type="button"
            onClick={() => onSelect(name)}
            className={`flex w-full items-center justify-between gap-2 rounded-lg px-3 py-2.5 text-left text-sm transition-colors ${
              isActive ? "bg-white/10 text-white" : "text-white/60 hover:bg-white/5"
            }`}
          >
            <span className="flex items-center gap-2">
              <Icon className="h-3.5 w-3.5" style={{ color: AGENT_HEX[kind].dark }} />
              {name}
            </span>
            <span className="font-mono text-xs text-white/50">{formatPercent(entry?.win_rate ?? null)}</span>
          </button>
        );
      })}
    </div>
  );
}

function EpisodeList({
  replays,
  selectedReplayId,
  onSelect,
}: {
  replays: ReplaySummary[];
  selectedReplayId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="max-h-[420px] overflow-y-auto border-b border-white/10 p-3 md:col-span-4 md:max-h-none md:border-r md:border-b-0">
      {replays.length === 0 && <p className="p-2 text-sm text-white/40">No recorded episodes for this agent yet.</p>}
      {replays.map((replay) => {
        const episodeNumber = replay.id.split("_").pop();
        const isActive = replay.id === selectedReplayId;
        return (
          <button
            key={replay.id}
            type="button"
            onClick={() => onSelect(replay.id)}
            className={`flex w-full items-center justify-between gap-2 rounded-lg px-3 py-2.5 text-left text-sm transition-colors ${
              isActive ? "bg-white/10" : "hover:bg-white/5"
            }`}
          >
            <span className="text-white/80">Episode {episodeNumber}</span>
            <span className="flex items-center gap-2 text-xs text-white/50">
              {replay.steps} moves
              {replay.won ? (
                <span className="flex items-center gap-1 text-emerald-300/80">
                  <Trophy className="h-3 w-3" /> Won
                </span>
              ) : (
                <span className="flex items-center gap-1 text-white/40">
                  <XCircle className="h-3 w-3" /> Lost
                </span>
              )}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function ReaderPane({
  agent,
  detail,
  status,
}: {
  agent: string;
  detail: ReplayDetail | null;
  status: "loading" | "success" | "error";
}) {
  const [stepIndex, setStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  const steps = useMemo(() => detail?.timeline ?? [], [detail]);

  useEffect(() => {
    setStepIndex(0);
    setIsPlaying(false);
  }, [detail?.id]);

  useEffect(() => {
    if (!isPlaying || steps.length === 0) return;
    if (stepIndex >= steps.length - 1) {
      setIsPlaying(false);
      return;
    }
    const timer = setTimeout(() => setStepIndex((i) => i + 1), 700 / speed);
    return () => clearTimeout(timer);
  }, [isPlaying, stepIndex, steps.length, speed]);

  if (status === "loading" || !detail) {
    return (
      <div className="p-5 md:col-span-5">
        <Skeleton className="h-72 w-full" />
      </div>
    );
  }

  const step = steps.length > 0 ? steps[stepIndex] : undefined;
  const board = step ? step.board_state : detail.initial_board;
  const mineHit = Boolean(step?.done) && !detail.won;
  const reason = step ? describeDecisionReason(agent, step.reasoning) : null;

  return (
    <div className="flex flex-col gap-4 p-5 md:col-span-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-sm font-medium text-white">
            {agent} · Episode {detail.episode_number ?? "—"}
          </p>
          <p className="text-xs text-white/50">
            {detail.board_size} board · {detail.mines} mines · {detail.won ? "won" : "lost"} in {detail.steps} moves
          </p>
        </div>
        {detail.won ? (
          <span className="flex items-center gap-1 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-2 py-1 text-[11px] text-emerald-300">
            <Trophy className="h-3 w-3" /> Win
          </span>
        ) : (
          <span className="flex items-center gap-1 rounded-full border border-white/10 bg-white/5 px-2 py-1 text-[11px] text-white/60">
            <XCircle className="h-3 w-3" /> Loss
          </span>
        )}
      </div>

      <div className="flex justify-center">
        <ReplayBoard
          board={board}
          highlightedCell={step ? { row: step.action.row, col: step.action.col } : undefined}
          mineHit={mineHit}
        />
      </div>

      <ReplayControls
        stepIndex={stepIndex}
        totalSteps={steps.length}
        isPlaying={isPlaying}
        speed={speed}
        onPrevious={() => setStepIndex((i) => Math.max(0, i - 1))}
        onNext={() => setStepIndex((i) => Math.min(steps.length - 1, i + 1))}
        onTogglePlay={() => setIsPlaying((p) => !p)}
        onSpeedChange={setSpeed}
      />

      <div className="liquid-glass flex items-start gap-2.5 rounded-lg p-3">
        <Sparkles className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[#A4F4FD]" />
        <div>
          <p className="text-xs font-medium tracking-wide text-white/70 uppercase">Why {agent} chose this cell</p>
          <p className="mt-1 text-sm text-white/70">
            {step ? (reason ?? "No reasoning recorded for this step.") : "Initial board — step forward to see its first move."}
          </p>
        </div>
      </div>
    </div>
  );
}
