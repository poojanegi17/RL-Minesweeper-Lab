import { motion } from "framer-motion";

/**
 * Purely decorative, purely visual -- no content, not a section. A few slow
 * ambient gradient blobs plus a faint grid texture behind the homepage's
 * section cards, so the page reads as one continuous lab space rather than
 * cards floating on flat black. `fixed` + `-z-10` so it stays put as the
 * page scrolls (an ambient backdrop, not something that moves with
 * content); mounted only inside `Home`, so it never bleeds into other
 * pages -- React unmounts it the moment the route changes.
 */
export function AmbientBackground() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden" aria-hidden="true">
      <div
        className="absolute inset-0 opacity-[0.05] dark:opacity-[0.07]"
        style={{
          backgroundImage:
            "linear-gradient(to right, var(--color-border) 1px, transparent 1px), linear-gradient(to bottom, var(--color-border) 1px, transparent 1px)",
          backgroundSize: "42px 42px",
        }}
      />

      <motion.div
        className="absolute -top-32 left-1/4 h-96 w-96 rounded-full bg-primary/10 blur-[100px]"
        animate={{ opacity: [0.5, 0.8, 0.5], x: [0, 20, 0] }}
        transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute top-1/3 -right-24 h-80 w-80 rounded-full bg-agent-dqn/10 blur-[100px]"
        animate={{ opacity: [0.4, 0.7, 0.4], y: [0, -24, 0] }}
        transition={{ duration: 22, repeat: Infinity, ease: "easeInOut", delay: 2 }}
      />
      <motion.div
        className="absolute bottom-24 left-1/3 h-72 w-72 rounded-full bg-agent-ppo/10 blur-[100px]"
        animate={{ opacity: [0.3, 0.6, 0.3] }}
        transition={{ duration: 20, repeat: Infinity, ease: "easeInOut", delay: 4 }}
      />
    </div>
  );
}
