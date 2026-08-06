import { motion } from "framer-motion";
import { Brain, Coins, Eye, Globe, RefreshCw, Target } from "lucide-react";
import { AlgorithmPipeline } from "@/components/agents/AlgorithmPipeline";
import { EngineeringChallengeCards } from "@/components/about/EngineeringChallengeCards";
import { SystemArchitectureDiagram } from "@/components/about/SystemArchitectureDiagram";
import { DesignPhilosophyProgression } from "@/components/about/DesignPhilosophyProgression";
import { TechStackGrid } from "@/components/about/TechStackGrid";
import { ProjectScaleMetrics } from "@/components/about/ProjectScaleMetrics";
import { RepositoryCallout } from "@/components/about/RepositoryCallout";

const AGENT_LOOP = [
  { title: "Environment", description: "The Minesweeper board and rules -- the one source of truth for what actually happens.", icon: Globe },
  { title: "Observation", description: "Hidden cells and revealed numbers, exactly as the agent receives them.", icon: Eye },
  { title: "Agent", description: "CSP, Q-Learning, DQN, or PPO -- decides what to do with that observation.", icon: Brain },
  { title: "Action", description: "One cell to reveal, chosen from everything still hidden.", icon: Target },
  { title: "Reward", description: "The environment scores the move -- win, lose, or reveal.", icon: Coins },
  { title: "Learning", description: "Agents that learn update their policy or value estimates from the outcome; others simply repeat the loop.", icon: RefreshCw },
];

export function About() {
  return (
    <div className="flex flex-col gap-24">
      {/* Hero */}
      <section className="grid items-center gap-12 lg:grid-cols-2">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className="flex flex-col items-start gap-6"
        >
          <span className="inline-flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1 font-mono text-[11px] tracking-wide text-text-muted uppercase">
            <span className="h-1.5 w-1.5 rounded-full bg-primary" />
            Behind the Lab
          </span>
          <h1 className="text-4xl leading-[1.05] font-semibold text-heading sm:text-5xl">Behind the Lab</h1>
          <p className="max-w-md text-lg text-text-muted">
            A reinforcement learning playground built to understand how different decision-making systems approach
            the same problem -- from deterministic reasoning to learned policies.
          </p>
        </motion.div>

        <div className="mx-auto w-full max-w-xs scale-90 sm:scale-100">
          <p className="mb-3 text-center text-xs font-medium tracking-wide text-text-muted uppercase">Every episode, one loop</p>
          <AlgorithmPipeline steps={AGENT_LOOP} accentClassName="text-primary" />
        </div>
      </section>

      {/* Engineering challenge */}
      <section>
        <SectionHeading title="Engineering challenge" />
        <p className="mb-8 max-w-2xl text-text-muted">
          Getting an agent to make one decision is the easy part. Making five different agents comparable, on the
          same board, with reproducible results, is most of the actual work.
        </p>
        <EngineeringChallengeCards />
      </section>

      {/* System architecture */}
      <section>
        <SectionHeading title="System architecture" />
        <p className="mb-8 max-w-2xl text-text-muted">
          Nothing on this site is mocked -- every page reads live from the same artifacts the RL pipeline writes.
          Hover a node below to see what it does.
        </p>
        <SystemArchitectureDiagram />
      </section>

      {/* Design philosophy */}
      <section>
        <SectionHeading title="Design philosophy" />
        <p className="mb-8 max-w-2xl text-text-muted">
          Five algorithms exist because each one answers a question the previous one couldn't.
        </p>
        <DesignPhilosophyProgression />
      </section>

      {/* Technical stack */}
      <section>
        <SectionHeading title="Technical stack" />
        <TechStackGrid />
      </section>

      {/* Project scale */}
      <section>
        <SectionHeading title="Project scale" />
        <ProjectScaleMetrics />
      </section>

      {/* Repository CTA */}
      <section>
        <RepositoryCallout />
      </section>
    </div>
  );
}

function SectionHeading({ title }: { title: string }) {
  return <h2 className="mb-3 text-sm font-medium tracking-wide text-text-muted uppercase">{title}</h2>;
}
