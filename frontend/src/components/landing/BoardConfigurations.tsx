import { useState } from "react";
import { Link } from "react-router-dom";
import { Check, Minus } from "lucide-react";
import { Skeleton } from "@/components/ui/Skeleton";
import { getLeaderboard } from "@/api/metrics";
import { getBoardConfigs } from "@/api/boardConfigs";
import { useApiQuery } from "@/hooks/useApiQuery";
import { formatPercent } from "@/lib/experimentAdapters";
import { shinyGradientStyle } from "@/lib/landingStyles";
import type { LeaderboardEntry } from "@/types/metrics";
import type { BoardLevelInfo } from "@/types/boardConfig";

const LEVELS = ["beginner", "intermediate", "expert"] as const;
type Level = (typeof LEVELS)[number];
const AGENT_ORDER = ["Random", "CSP", "Q-Learning", "DQN", "PPO"];
const EVAL_EPISODES = 2000;

const LEVEL_LABEL: Record<Level, string> = { beginner: "Beginner", intermediate: "Intermediate", expert: "Expert" };
const LEVEL_TAGLINE: Record<Level, string> = {
  beginner: "Where every agent starts.",
  intermediate: "Where generalization starts to break.",
  expert: "The hardest board this lab evaluates.",
};

interface ConfigsData {
  boardConfigs: BoardLevelInfo[];
  leaderboards: Record<string, LeaderboardEntry[]>;
}

function bestEntry(entries: LeaderboardEntry[]): LeaderboardEntry | null {
  const trained = entries.filter((e) => e.win_rate != null);
  if (trained.length === 0) return null;
  return trained.reduce((best, e) => (e.win_rate! > best.win_rate! ? e : best));
}

/**
 * The pricing-section equivalent: three board configurations instead of
 * three plans, the yearly/monthly toggle repurposed as sparse/dense mine
 * density (never "standard" -- see the comment below on why that density is
 * deliberately excluded), and every card's numbers sourced live from
 * `/api/leaderboard` rather than authored copy.
 */
export function BoardConfigurations() {
  const [dense, setDense] = useState(false);
  const density = dense ? "dense" : "sparse";

  const { data, status } = useApiQuery<ConfigsData>(async () => {
    const boardConfigs = await getBoardConfigs();
    const pairs = await Promise.all(
      LEVELS.flatMap((level) =>
        (["sparse", "dense"] as const).map(async (d) => {
          const entries = await getLeaderboard(level, d);
          return [`${level}-${d}`, entries] as const;
        }),
      ),
    );
    return { boardConfigs, leaderboards: Object.fromEntries(pairs) };
  }, []);

  return (
    <section className="lp-bc-section">
      <div className="lp-bc-watermark">
        <div className="lp-bc-watermark-main">
          <span className="lp-bc-watermark-line-1">Same agents.</span>
          <span className="lp-bc-watermark-line-2">Different boards.</span>
        </div>
      </div>

      <p className="relative z-[3] mt-6 max-w-lg text-center text-sm text-white/50">
        Every agent is evaluated on three board sizes, at every mine density, over{" "}
        {EVAL_EPISODES.toLocaleString()} episodes each — not just the one config it happened to train on.
      </p>

      {status === "loading" && (
        <div className="relative z-[3] mt-16 grid w-full max-w-[1100px] grid-cols-1 gap-6 md:grid-cols-3">
          <Skeleton className="h-[580px] w-full rounded-[44px]" />
          <Skeleton className="h-[580px] w-full rounded-[44px]" />
          <Skeleton className="h-[580px] w-full rounded-[44px]" />
        </div>
      )}

      {data && (
        <div className="lp-bc-grid">
          {LEVELS.map((level) => {
            const boardConfig = data.boardConfigs.find((c) => c.level === level);
            const entries = data.leaderboards[`${level}-${density}`] ?? [];
            const best = bestEntry(entries);
            const mines = boardConfig?.densities[density];

            return (
              <div key={level} className={`lp-bc-card${level === "expert" ? " lp-bc-card-expert" : ""}`}>
                <p className="lp-bc-tier-small">{LEVEL_LABEL[level]}</p>
                <p className="lp-bc-tier-large" style={best ? shinyGradientStyle : undefined}>
                  {best ? formatPercent(best.win_rate) : "—"}
                </p>
                <p className="mt-1 text-xs text-white/40">
                  {best ? `best result: ${best.agent}` : "no results yet at this density"}
                </p>
                <p className="lp-bc-desc">
                  {boardConfig ? `${boardConfig.rows}×${boardConfig.cols} board, ${mines} mines. ` : ""}
                  {LEVEL_TAGLINE[level]} Evaluated over {EVAL_EPISODES.toLocaleString()} episodes, seed 42.
                </p>
                <ul className="lp-bc-list">
                  {AGENT_ORDER.map((agent) => {
                    const entry = entries.find((e) => e.agent === agent);
                    const trained = entry?.win_rate != null;
                    return (
                      <li key={agent}>
                        <span className="lp-bc-check">
                          {trained ? (
                            <Check className="h-3.5 w-3.5 text-white" />
                          ) : (
                            <Minus className="h-3.5 w-3.5 text-white/30" />
                          )}
                        </span>
                        <span>
                          {agent} — {trained ? formatPercent(entry!.win_rate) : "not yet trained"}
                        </span>
                      </li>
                    );
                  })}
                </ul>
                <Link to="/compare" className="lp-bc-btn">
                  Explore this board
                </Link>
              </div>
            );
          })}
        </div>
      )}

      <div className="lp-bc-toggle-wrap">
        <span className="text-sm text-white/70">Dense mines</span>
        <button
          type="button"
          onClick={() => setDense((v) => !v)}
          className={`lp-bc-toggle${dense ? " active" : ""}`}
          role="switch"
          aria-checked={dense}
          aria-label="Toggle dense mine density"
        >
          <span className="lp-bc-toggle-knob" />
        </button>
      </div>
    </section>
  );
}
