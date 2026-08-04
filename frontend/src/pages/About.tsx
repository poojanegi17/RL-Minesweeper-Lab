import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/cn";

const STACK = [
  "React",
  "TypeScript",
  "Vite",
  "Tailwind CSS",
  "React Router",
  "Framer Motion",
  "Recharts",
  "Lucide Icons",
];

const ROADMAP = [
  {
    title: "V1 — this build",
    description:
      "Static frontend: agent catalog, architecture explanations, and a comparison dashboard, all backed by mock data.",
    current: true,
  },
  {
    title: "V2 — playable game & replay",
    description:
      "A human-playable Minesweeper board, plus interactive replay of a trained agent's decisions move by move.",
    current: false,
  },
  {
    title: "V3 — real training data",
    description:
      "Swap mock data for a real backend: live training reports, learning curves, and experiment tracking.",
    current: false,
  },
];

export function About() {
  return (
    <div className="flex flex-col gap-12">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight text-heading">
          About this project
        </h1>
        <p className="mt-3 max-w-2xl leading-relaxed text-text-muted">
          RL Minesweeper Lab is an educational, research-oriented platform for
          comparing how different approaches — a rule-based solver and
          reinforcement learning agents — reason about the same problem:
          playing Minesweeper. It's built to demonstrate machine learning
          engineering, reinforcement learning, and frontend product design in
          one place, not to be a Minesweeper game on its own.
        </p>
      </div>

      <div>
        <h2 className="mb-3 text-sm font-medium tracking-wide text-text-muted uppercase">
          Built with
        </h2>
        <div className="flex flex-wrap gap-2">
          {STACK.map((item) => (
            <span
              key={item}
              className="rounded-full border border-border bg-surface px-3 py-1 font-mono text-sm text-text"
            >
              {item}
            </span>
          ))}
        </div>
      </div>

      <div>
        <h2 className="mb-5 text-sm font-medium tracking-wide text-text-muted uppercase">
          Roadmap
        </h2>
        <ol className="flex flex-col">
          {ROADMAP.map((phase, index) => (
            <li key={phase.title} className="flex gap-4">
              <div className="flex flex-col items-center">
                <span
                  className={cn(
                    "flex h-7 w-7 shrink-0 items-center justify-center rounded-full font-mono text-xs font-medium",
                    phase.current
                      ? "bg-primary text-primary-foreground"
                      : "bg-surface text-text-muted",
                  )}
                >
                  {index + 1}
                </span>
                {index < ROADMAP.length - 1 && (
                  <span className="my-1 w-px flex-1 bg-border" />
                )}
              </div>
              <Card
                className={cn(
                  "mb-4 flex-1",
                  phase.current && "border-primary/40",
                )}
              >
                <div className="flex items-center gap-2">
                  <h3 className="font-semibold text-heading">
                    {phase.title}
                  </h3>
                  {phase.current && (
                    <span className="rounded-full bg-primary/10 px-2 py-0.5 font-mono text-[10px] tracking-wide text-primary uppercase">
                      You are here
                    </span>
                  )}
                </div>
                <p className="mt-1 text-sm text-text-muted">
                  {phase.description}
                </p>
              </Card>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
