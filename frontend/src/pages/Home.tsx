import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, GitCompare, Info, Layers } from "lucide-react";
import { BoardIllustration } from "@/components/board/BoardIllustration";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { getAgents } from "@/data/agents";

const EXPLORE_LINKS = [
  {
    to: "/agents",
    icon: Layers,
    title: "Agent catalog",
    description:
      "Browse each solving approach — from deterministic logic to policy-gradient reinforcement learning — with its architecture and metrics.",
  },
  {
    to: "/compare",
    icon: GitCompare,
    title: "Compare dashboard",
    description:
      "Put two agents side by side and see how their performance stacks up.",
  },
  {
    to: "/about",
    icon: Info,
    title: "About the project",
    description:
      "The motivation behind this lab, the tech stack, and where it's headed next.",
  },
];

export function Home() {
  const agents = getAgents();
  const trainedCount = agents.filter((a) => a.status === "trained").length;
  const bestWinRate = Math.max(
    ...agents.flatMap((a) =>
      a.metrics.filter((m) => m.label === "Win Rate").map((m) => m.value),
    ),
  );

  const stats = [
    { label: "Agents documented", value: agents.length },
    { label: "Fully trained", value: trainedCount },
    { label: "Best win rate", value: `${bestWinRate}%` },
  ];

  return (
    <div className="flex flex-col gap-20">
      <section className="grid items-center gap-12 md:grid-cols-2">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className="flex flex-col items-start gap-6"
        >
          <span className="inline-flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1 font-mono text-[11px] tracking-wide text-text-muted uppercase">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" />
            Reinforcement learning research
          </span>

          <h1 className="text-5xl leading-[1.05] font-semibold text-heading">
            RL Minesweeper Lab
          </h1>
          <p className="max-w-md text-lg text-text-muted">
            An educational platform comparing how a rule-based solver and
            three reinforcement learning agents — Q-Learning, DQN, and PPO —
            approach the same game of Minesweeper.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link to="/agents">
              <Button>
                Explore agents
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
            <Link to="/compare">
              <Button variant="secondary">Compare performance</Button>
            </Link>
          </div>

          <dl className="mt-2 grid grid-cols-3 gap-6 border-t border-border pt-6">
            {stats.map((stat) => (
              <div key={stat.label}>
                <dd className="font-mono text-2xl font-semibold text-heading">
                  {stat.value}
                </dd>
                <dt className="mt-1 text-xs text-text-muted">{stat.label}</dt>
              </div>
            ))}
          </dl>
        </motion.div>

        <div className="relative flex justify-center md:justify-end">
          <div
            className="absolute inset-0 -z-10 rounded-full bg-primary/10 blur-3xl"
            aria-hidden="true"
          />
          <BoardIllustration />
        </div>
      </section>

      <section>
        <div className="mb-6 flex items-baseline justify-between">
          <h2 className="text-sm font-medium tracking-wide text-text-muted uppercase">
            Explore
          </h2>
        </div>
        <div className="grid gap-4 sm:grid-cols-3">
          {EXPLORE_LINKS.map((item) => (
            <Link key={item.to} to={item.to} className="block h-full">
              <Card interactive className="flex h-full flex-col gap-4">
                <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <item.icon className="h-4 w-4" />
                </span>
                <div>
                  <h3 className="font-semibold text-heading">{item.title}</h3>
                  <p className="mt-1 text-sm text-text-muted">
                    {item.description}
                  </p>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
