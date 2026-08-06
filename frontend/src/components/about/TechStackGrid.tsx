import { motion, type Variants } from "framer-motion";
import { Card } from "@/components/ui/Card";

const GROUPS = [
  { label: "Frontend", items: ["React", "TypeScript", "Vite", "Tailwind CSS", "React Router", "Framer Motion", "Recharts"] },
  { label: "Backend", items: ["FastAPI", "Python"] },
  { label: "Reinforcement Learning", items: ["Gymnasium", "PyTorch", "CSP", "Q-Learning", "DQN", "PPO"] },
  { label: "Infrastructure", items: ["Experiment tracking", "Evaluation pipeline", "Replay system"] },
];

const containerVariants: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.08 } },
};
const cardVariants: Variants = {
  hidden: { opacity: 0, y: 12 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.35, ease: "easeOut" } },
};

/** The stack, grouped by layer instead of one flat badge list -- makes it
 * legible at a glance which layer each technology belongs to. */
export function TechStackGrid() {
  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, amount: 0.2 }}
      className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"
    >
      {GROUPS.map((group) => (
        <motion.div key={group.label} variants={cardVariants}>
          <Card className="flex h-full flex-col gap-3 border-white/10 bg-gradient-to-b from-surface/80 to-surface/50 shadow-md shadow-black/[0.05] backdrop-blur-sm">
            <h3 className="text-xs font-medium tracking-wide text-text-muted uppercase">{group.label}</h3>
            <div className="flex flex-wrap gap-2">
              {group.items.map((item) => (
                <motion.span
                  key={item}
                  whileHover={{ scale: 1.06, y: -1 }}
                  transition={{ duration: 0.15 }}
                  className="rounded-full border border-border bg-surface px-2.5 py-1 font-mono text-xs text-text-muted transition-colors hover:border-primary/40 hover:text-primary"
                >
                  {item}
                </motion.span>
              ))}
            </div>
          </Card>
        </motion.div>
      ))}
    </motion.div>
  );
}
