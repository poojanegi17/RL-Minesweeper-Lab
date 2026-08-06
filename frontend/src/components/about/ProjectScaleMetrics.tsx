import { motion, type Variants } from "framer-motion";
import { Card } from "@/components/ui/Card";
import { ApiErrorState } from "@/components/ui/ApiErrorState";
import { Skeleton } from "@/components/ui/Skeleton";
import { getAgents } from "@/api/agents";
import { getExperiments } from "@/api/experiments";
import { getReplays } from "@/api/replays";
import { useApiQuery } from "@/hooks/useApiQuery";

/** Real `def test_...` functions under `backend/tests/*.py` as of this
 * writing. There's no API exposing a live test count (the backend is a
 * read-only artifact server, not a CI dashboard) -- shown as a static,
 * verifiable project fact, the same way the tech stack list below is a
 * static real fact rather than a live API response. */
const BACKEND_TEST_COUNT = 88;

interface ScaleData {
  algorithmsTested: number;
  experimentsPerformed: number;
  trainingEpisodes: number;
  recordedReplays: number;
}

async function fetchScaleData(): Promise<ScaleData> {
  const [agents, experiments, replays] = await Promise.all([getAgents(), getExperiments(), getReplays()]);

  const experimentsPerformed = experiments.reduce((sum, e) => sum + e.run_count, 0);
  const trainingEpisodes = experiments.reduce((sum, e) => sum + e.runs.reduce((s, r) => s + r.episodes, 0), 0);

  return {
    algorithmsTested: agents.length,
    experimentsPerformed,
    trainingEpisodes,
    recordedReplays: replays.length,
  };
}

const containerVariants: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.08 } },
};
const cardVariants: Variants = {
  hidden: { opacity: 0, y: 12 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.35, ease: "easeOut" } },
};

/**
 * Portfolio-scale figures, computed from the same real API responses every
 * other page reads -- never hardcoded, except the one figure with no API to
 * source it from (backend test count), which is clearly labeled as such.
 */
export function ProjectScaleMetrics() {
  const { data, status, error, retry } = useApiQuery(fetchScaleData, []);

  if (status === "loading") {
    return (
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5" aria-label="Loading project metrics">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-24 w-full" />
        ))}
      </div>
    );
  }

  if (status === "error" && error) {
    return <ApiErrorState error={error} onRetry={retry} title="Couldn't load project metrics" />;
  }

  const stats = [
    { label: "Algorithms tested", value: data?.algorithmsTested ?? null },
    { label: "Experiments performed", value: data?.experimentsPerformed ?? null },
    { label: "Training episodes", value: data?.trainingEpisodes ?? null },
    { label: "Recorded replays", value: data?.recordedReplays ?? null },
    { label: "Backend tests", value: BACKEND_TEST_COUNT },
  ];

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.3 }}
      className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5"
    >
      {stats.map((stat) => (
        <motion.div key={stat.label} variants={cardVariants}>
          <Card className="flex flex-col gap-1 border-white/10 bg-gradient-to-b from-surface/80 to-surface/50 text-center shadow-md shadow-black/[0.05] backdrop-blur-sm">
            <span className="font-mono text-2xl font-semibold text-heading tabular-nums">
              {stat.value != null ? stat.value.toLocaleString() : "—"}
            </span>
            <span className="text-xs text-text-muted">{stat.label}</span>
          </Card>
        </motion.div>
      ))}
    </motion.div>
  );
}
