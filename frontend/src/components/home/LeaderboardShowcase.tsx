import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { AGENT_ICONS } from "@/components/agent/agentIcons";
import { useTheme } from "@/app/ThemeProvider";
import { agentKindFromName, slugifyAgentName } from "@/lib/agentAdapters";
import { formatPercent } from "@/lib/experimentAdapters";
import { AGENT_HEX, AGENT_STYLES } from "@/data/types";
import type { LeaderboardEntry } from "@/types/metrics";
import { cn } from "@/lib/cn";

interface LeaderboardShowcaseProps {
  /** Already ranked by the backend (`GET /api/leaderboard`, best win rate
   * first) -- never re-sorted or re-derived here. */
  leaderboard: LeaderboardEntry[];
}

/** An animated win-rate bar per agent, in place of the plain ranking table
 * elsewhere in the app -- same real `GET /api/leaderboard` data, just a
 * more visual read for a landing page. */
export function LeaderboardShowcase({ leaderboard }: LeaderboardShowcaseProps) {
  const { theme } = useTheme();
  const maxWinRate = Math.max(...leaderboard.map((entry) => entry.win_rate ?? 0), 0.01);

  return (
    <div className="flex flex-col gap-4">
      <p className="max-w-2xl text-sm text-text-muted">Different approaches fail for different reasons.</p>

      <div className="flex flex-col gap-3">
        {leaderboard.map((entry, index) => {
          const kind = agentKindFromName(entry.agent);
          const style = AGENT_STYLES[kind];
          const Icon = AGENT_ICONS[kind];
          const barPercent = entry.win_rate != null ? Math.max(4, (entry.win_rate / maxWinRate) * 100) : 0;

          return (
            <motion.div
              key={entry.agent}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05, ease: "easeOut" }}
            >
              <Card interactive className="flex items-center gap-4">
                <span className="w-5 shrink-0 text-center font-mono text-sm text-text-muted">{index + 1}</span>
                <span className={cn("flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-current/10", style.text)}>
                  <Icon className="h-4 w-4" />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <Link to={`/agents/${slugifyAgentName(entry.agent)}`} className="truncate font-semibold text-heading hover:text-primary">
                      {entry.agent}
                    </Link>
                    <span className="shrink-0 font-mono text-sm font-medium text-heading">{formatPercent(entry.win_rate)}</span>
                  </div>
                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-surface-hover">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ background: AGENT_HEX[kind][theme] }}
                      initial={{ width: 0 }}
                      animate={{ width: `${barPercent}%` }}
                      transition={{ duration: 0.6, ease: "easeOut", delay: 0.1 + index * 0.05 }}
                    />
                  </div>
                </div>
                <Badge variant="outline" className="shrink-0">
                  {entry.source === "experiment_artifact" ? "Live" : "Reference"}
                </Badge>
              </Card>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
