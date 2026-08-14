import { AlertTriangle, MousePointerClick, ShieldCheck } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/lib/cn";

interface EnvRow {
  agent: string;
  v1: number | null;
  v2: number | null;
  /** Set where the delta needs reading differently from the rest of the column. */
  note?: string;
}

/**
 * Beginner/standard (5x5, 5 mines), 2,000 greedy episodes at seed 42, mirroring
 * `results_public/{v1,v2}/levels/beginner/standard/*_board_result.json`. The
 * Q-Learning row is its shipped 100,000-episode budget.
 */
const BEGINNER: EnvRow[] = [
  { agent: "Random", v1: 0.0045, v2: 0.013, note: "Learns nothing — so this gap is the game getting easier, not an agent improving." },
  { agent: "CSP", v1: 0.434, v2: 0.7035, note: "A fixed solver with no weights. It gains 26.95 points from the rule change alone." },
  { agent: "Q-Learning", v1: 0.019, v2: 0.717 },
  { agent: "DQN", v1: 0.3855, v2: 0.7725 },
  { agent: "PPO", v1: 0.0125, v2: 0.0795 },
];

/** Intermediate/standard (9x9, 12 mines), same protocol. */
const INTERMEDIATE: EnvRow[] = [
  { agent: "CSP", v1: 0.609, v2: 0.8995 },
  { agent: "DQN", v1: 0.5275, v2: 0.8015 },
  { agent: "PPO", v1: 0, v2: 0, note: "The one place the change bought nothing at all." },
];

function pct(value: number | null): string {
  return value == null ? "—" : `${(value * 100).toFixed(2)}%`;
}

function ResultTable({ caption, rows }: { caption: string; rows: EnvRow[] }) {
  return (
    <div>
      <p className="mb-2 text-xs font-medium tracking-wide text-text-muted uppercase">{caption}</p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[380px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-border text-left text-xs text-text-muted uppercase">
              <th className="pb-2 pr-3 font-medium">Agent</th>
              <th className="pb-2 pr-3 font-medium">Original</th>
              <th className="pb-2 pr-3 font-medium">Safe opening</th>
              <th className="pb-2 font-medium">Change</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const delta = row.v1 != null && row.v2 != null ? (row.v2 - row.v1) * 100 : null;
              const flat = delta != null && Math.abs(delta) < 0.005;
              return (
                <tr key={row.agent} className="border-b border-border last:border-0 align-top">
                  <td className="py-2 pr-3 text-text">{row.agent}</td>
                  <td className="py-2 pr-3 font-mono text-text-muted">{pct(row.v1)}</td>
                  <td className="py-2 pr-3 font-mono font-medium text-heading">{pct(row.v2)}</td>
                  <td className={cn("py-2 font-mono", flat ? "text-text-muted" : "text-emerald-500")}>
                    {delta == null ? "—" : flat ? "none" : `+${delta.toFixed(2)}`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <ul className="mt-2 flex flex-col gap-1">
        {rows
          .filter((row) => row.note)
          .map((row) => (
            <li key={row.agent} className="text-xs text-text-muted">
              <span className="text-text">{row.agent}:</span> {row.note}
            </li>
          ))}
      </ul>
    </div>
  );
}

/**
 * The two board distributions every result in this project is measured under,
 * why the second one exists, and what it actually bought.
 *
 * This card exists because the single most misreadable thing on the site is a
 * v1 number sitting next to a v2 number. They are not two difficulty settings
 * of one game -- under the original rules the opening click can be a mine, so
 * roughly a fifth of 5x5 episodes are decided before any agent acts. Subtracting
 * across the two columns is the error the card is built to prevent, which is why
 * the Random and CSP rows are given first: neither learns anything, and both
 * gain, so part of every other row's gain is the benchmark getting easier rather
 * than the agent getting better.
 */
export function EnvironmentsTested() {
  return (
    <div className="flex flex-col gap-6">
      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="flex flex-col gap-3">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-surface-hover text-text-muted">
              <MousePointerClick className="h-4 w-4" />
            </span>
            <div>
              <h3 className="font-semibold text-heading">Original environment</h3>
              <p className="font-mono text-xs text-text-muted">first_click_safe: none</p>
            </div>
          </div>
          <p className="text-sm text-text-muted">
            Mines are placed before the first click, so the opening move can lose the game outright. This is the
            environment the whole project started on, and every early conclusion was drawn under it.
          </p>
          <p className="text-sm text-text-muted">
            Its problem is that roughly a fifth of 5×5 episodes end before the agent has made a real decision. Those
            losses are unwinnable by construction, and the gradient they produce is pure noise.
          </p>
        </Card>

        <Card className="flex flex-col gap-3">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <ShieldCheck className="h-4 w-4" />
            </span>
            <div>
              <h3 className="font-semibold text-heading">Safe-opening environment</h3>
              <p className="font-mono text-xs text-text-muted">first_click_safe: area</p>
            </div>
          </div>
          <p className="text-sm text-text-muted">
            The 3×3 block around the opening click is kept mine-free, so no game is lost on move one and every episode
            begins from a cascade — the rule most real Minesweeper implementations use.
          </p>
          <p className="text-sm text-text-muted">
            Forced guesses still exist later in the game, deliberately. That keeps the CSP solver a meaningful ceiling
            rather than a trivial 100%, so the benchmark still has something to measure.
          </p>
        </Card>
      </div>

      <Card className="flex flex-col gap-5">
        <div>
          <h3 className="font-semibold text-heading">Did it help?</h3>
          <p className="mt-1 text-sm text-text-muted">
            Yes — and for two different reasons that have to be separated, which is what the first two rows are for.
          </p>
        </div>

        <ResultTable caption="Beginner — 5×5, 5 mines" rows={BEGINNER} />
        <ResultTable caption="Intermediate — 9×9, 12 mines" rows={INTERMEDIATE} />

        <div className="flex flex-col gap-2 border-t border-border pt-4">
          <p className="text-sm text-text-muted">
            <span className="text-text">Part of every gain is the game, not the agent.</span> Random and CSP consult no
            reward and have no weights, yet both improve — so a protected opening is simply an easier game before any
            learning difference is considered. A fair reading of DQN's jump compares it against its own v1 model
            re-scored on the new benchmark (54.90%), not against its v1 score, which puts the training gain at 22.35
            points rather than 38.70.
          </p>
          <p className="text-sm text-text-muted">
            <span className="text-text">The exception is the most informative row.</span> PPO at 9×9 gains nothing:
            0.00% either way, on identical boards. The change demonstrably applied — first-move losses fell from 2,908
            per 20,000 training episodes to 1 — so an entire failure mode was removed and the win rate did not move.
            That is evidence the opening was never what limited PPO on a larger board; a win there needs 69 correct
            reveals and its median episode ends after 4.
          </p>
        </div>

        <div className="flex items-start gap-2 rounded-lg border border-amber-500/40 bg-amber-500/5 px-3 py-2">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" aria-hidden="true" />
          <p className="text-xs text-text-muted">
            <span className="text-text">These are two different games, not two difficulty levels.</span> A win rate
            under one cannot be subtracted from a win rate under the other. They are stored in separate result trees and
            every file records its own <span className="font-mono">env_version</span>, so nothing on this site mixes
            them — the environment control on each page picks one and reads it end to end.
          </p>
        </div>

        <div className="flex flex-wrap gap-1.5">
          <Badge variant="outline">2,000 episodes per cell</Badge>
          <Badge variant="outline">seed 42</Badge>
          <Badge variant="outline">identical boards within a cell</Badge>
        </div>
      </Card>
    </div>
  );
}
