import { motion } from "framer-motion";
import { Skeleton } from "@/components/ui/Skeleton";
import { WheelCarousel, type WheelItemState } from "@/components/ui/WheelCarousel";
import { headlineEyebrowClass, staticHeadlineGradient } from "@/lib/landingStyles";
import { cn } from "@/lib/cn";
import { getLeaderboard } from "@/api/metrics";
import { useApiQuery } from "@/hooks/useApiQuery";
import { formatPercent } from "@/lib/experimentAdapters";
import type { LeaderboardEntry } from "@/types/metrics";

function winRate(board: LeaderboardEntry[] | undefined, agent: string): number | null {
  return board?.find((e) => e.agent === agent)?.win_rate ?? null;
}

interface FindingsData {
  protectedStandard: LeaderboardEntry[];
}

/**
 * The testimonials-slot equivalent -- same quote-card layout, but every
 * "quote" is a factual statement about this project's own measurements, never
 * a fabricated person or company, and attributed to the runs it came from
 * instead of a name and job title.
 *
 * These are deliberately the *surprising* results rather than the flattering
 * ones -- several are things the project expected to go the other way.
 *
 * Every card must point at something a visitor can actually go and find. That
 * rules out the bootstrap-bug story, which is the single most striking result
 * in the project's history but concerns five runs `HIDDEN_VARIANTS` keeps off
 * the site entirely: a card claiming "five experiments were measuring a bug"
 * sends a reader looking for experiments no page renders. It stays in the DQN
 * chapter, where the corrected baseline it produced is the headline run.
 *
 * Cards are not clickable, so each states its finding in full rather than
 * teasing a detail page. Numbers are exact and quoted with their significance
 * where a comparison is being claimed; where a difference is *not* significant
 * the card says so rather than implying it.
 *
 * The first two cards read live from the leaderboard, so they cannot drift from
 * what the API serves. The rest quote measurements that live outside it --
 * cross-benchmark evaluations, zero-shot transfer runs, and a historical
 * comparison against an implementation that no longer exists -- and are sourced
 * in `PIPELINE_CHAPTERS` / `VARIANT_LEVEL_CONCLUSIONS`, where each is derived
 * from the artifacts under `rl/results_public/`.
 */
export function FindingsGrid() {
  const { data, status } = useApiQuery<FindingsData>(async () => {
    // The protected 5x5 benchmark, where the opening click cannot lose -- the
    // distribution both surprises below were measured on.
    const protectedStandard = await getLeaderboard("beginner", "standard", "area");
    return { protectedStandard };
  }, []);

  const findings = data && [
    {
      quote: `A neural network beat the deduction solver. On boards where the opening click is guaranteed safe, DQN wins ${formatPercent(winRate(data.protectedStandard, "DQN"))} against CSP's ${formatPercent(winRate(data.protectedStandard, "CSP"))} — and CSP is not a heuristic, it is explicit constraint propagation that proves cells safe. For most of this project the working assumption was that no learned agent would pass it.`,
      attribution: "DQN vs. CSP — the assumption that broke",
      context: "5×5 · 5 mines · first click safe · 2,000 episodes",
    },
    {
      quote: `A lookup table with no generalization whatsoever draws level with that same solver: Q-Learning reaches ${formatPercent(winRate(data.protectedStandard, "Q-Learning"))} against CSP's ${formatPercent(winRate(data.protectedStandard, "CSP"))} (p = 0.35, statistically indistinguishable). It cannot recognise a board it has not seen before — it simply sees most of them, because a protected 5×5 opening repeats often enough to memorise.`,
      attribution: "Q-Learning — memorisation is enough at 5×5",
      context: "5×5 · 5 mines · first click safe · 100,000 training episodes",
    },
    {
      quote:
        "For PPO at 9×9, training was worth nothing. Loading its 5×5 weights straight into a 9×9 game with no retraining at all scores 1.05% at sparse density, against 0.70% for the same recipe trained there for 100,000 episodes — nominally ahead, though at p = 0.31 that gap is not one this evidence can carry. The honest version is that 100,000 episodes bought no measurable improvement over weights carried from a smaller board. DQN's equivalent test loses roughly half its win rate without retraining (80.15% to 40.10%), which is what a functioning agent looks like on the same measurement.",
      attribution: "PPO — zero-shot transfer matches retraining",
      context: "9×9 · 8 mines · first click safe · 2,000 episodes",
    },
    {
      quote:
        "PPO is the wrong tool for this board, and the comparison is now airtight. On boards where a correct move always exists, PPO wins 13.30% and DQN wins 99.65% — with the same 11-channel encoding, the same 100,000-episode budget, comparable gradient updates, and, since the fully-convolutional network was tested for both, the same architecture. Every hyperparameter changed on PPO came back null; only changing the board ever moved it. The likely cause is structural: PPO discards each rollout after a few gradient steps, while DQN keeps 20,000 transitions and re-uses every rare win thousands of times.",
      attribution: "PPO vs. DQN — a structural mismatch, not a tuning gap",
      context: "5×5 · solvable boards · matched compute · 2,000 episodes",
    },
    {
      quote:
        "Three optimization settings took DQN from 21.25% to 37.90% — and two of them are actively harmful on their own. Applied without the third (a reduced replay ratio), reward scaling and a longer exploration schedule together score 5.95%, far worse than changing nothing at all. Their individual contributions sum to 48.85 points against a bundle worth 16.65, because none of them is worth much until the other two are already in place. Reported separately, any one of the three would have looked like a failure.",
      attribution: "DQN — three fixes that only work together",
      context: "5×5 · 5 mines · unprotected boards · 100,000 episodes each",
    },
    {
      quote:
        "Under-spending compute did not shrink two findings, it reversed their sign. An earlier version of PPO's environment chapter ran at roughly 25,000 gradient updates instead of 123,000 and concluded that no-guess training boards were mildly helpful and that the discount factor did not matter. Re-run with the budget matched, no-guess boards are a real loss and the discount factor costs 4.85 points. Both original conclusions were confidently wrong in the opposite direction, which is the most transferable lesson here: verify you have converged before you interpret an ablation.",
      attribution: "Methodology — two conclusions that inverted",
      context: "5×5 · PPO · 25,000 vs. 123,000 gradient updates",
    },
  ];

  return (
    <section className="relative z-10 mx-auto flex min-h-screen max-w-6xl flex-col items-center justify-center border-t border-white/10 px-6 py-12">
      <motion.h2
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7 }}
        className="text-center text-5xl leading-[0.95] font-semibold tracking-tight md:text-6xl"
      >
        <span className={headlineEyebrowClass}>06 findings · 2,000 episodes each</span>
        <span className="block text-white">What</span>
        <span className="block" style={staticHeadlineGradient}>
          surprised us
        </span>
      </motion.h2>

      <motion.p
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7, delay: 0.1 }}
        className="mt-4 text-center text-xs text-white/50"
      >
        Six results we did not expect — hover a card to read it in full.
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.7, delay: 0.2 }}
        // Extra gap, not decoration: the detail panel is taller than the
        // stage and overhangs it, so this is the clearance that keeps it from
        // ever creeping up on the heading whichever card is open.
        className="mt-12 w-full"
      >
        {status === "loading" && (
          <div className="flex justify-center">
            <Skeleton className="h-[300px] w-full max-w-[230px] rounded-2xl" />
          </div>
        )}

        {findings && (
          <WheelCarousel<Finding>
            items={findings}
            getKey={(finding) => finding.attribution}
            ariaLabel="What surprised us"
            cardWidth={230}
            cardHeight={300}
            renderItem={(finding, state) => <FindingCard finding={finding} state={state} />}
            renderOverlay={(finding) => <FindingDetail finding={finding} />}
          />
        )}
      </motion.div>
    </section>
  );
}

interface Finding {
  quote: string;
  attribution: string;
  context: string;
}

/** The card itself is unchanged from the flat grid it replaced -- same
 * `liquid-glass` surface, same blockquote/figcaption structure, same
 * typography. Only the active treatment is new, and it is additive. */
function FindingCard({ finding, state }: { finding: Finding; state: WheelItemState }) {
  return (
    <figure
      className={cn(
        "liquid-glass lp-wheel-card flex h-full flex-col justify-center rounded-2xl p-4 text-center transition-colors duration-500",
        state.isActive && "lp-wheel-card--active",
      )}
    >
      <p className="text-sm leading-snug font-semibold text-white">{finding.attribution}</p>
      <p className="mt-2 text-[10px] font-semibold tracking-wide text-white/45 uppercase">{finding.context}</p>
    </figure>
  );
}

/** The active finding, floated over the wheel by `WheelCarousel`'s overlay
 * slot.
 *
 * Narrow and tall on purpose: it is the active *card* expanding, so it keeps
 * roughly a card's width and grows vertically past the wheel instead of
 * spreading sideways across it.
 *
 * It is also opaque rather than glass. Every other surface on this page is
 * see-through, but this one sits directly on top of six lit cards -- and glass
 * over a bright, busy backdrop is simply unreadable. Legibility wins over
 * consistency here. */
function FindingDetail({ finding }: { finding: Finding }) {
  return (
    <div className="lp-wheel-panel w-[340px] max-w-[88vw] rounded-2xl p-6">
      <p className="text-sm leading-snug font-semibold text-white">{finding.attribution}</p>
      <p className="mt-1.5 text-[10px] font-semibold tracking-wide text-white/45 uppercase">{finding.context}</p>
      <p className="mt-4 border-t border-white/10 pt-4 text-xs leading-relaxed text-white/85">{finding.quote}</p>
    </div>
  );
}
