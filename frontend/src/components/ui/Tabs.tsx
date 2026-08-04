import { useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { cn } from "@/lib/cn";

interface Tab {
  id: string;
  label: string;
}

interface TabsProps {
  tabs: Tab[];
  defaultTab?: string;
  children: (activeTab: string) => ReactNode;
}

export function Tabs({ tabs, defaultTab, children }: TabsProps) {
  const [active, setActive] = useState(defaultTab ?? tabs[0]?.id);

  return (
    <div>
      <div className="flex gap-1 border-b border-border">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActive(tab.id)}
            className="relative -mb-px px-4 py-2.5 text-sm font-medium transition-colors"
          >
            <span
              className={cn(
                active === tab.id
                  ? "text-heading"
                  : "text-text-muted hover:text-text",
              )}
            >
              {tab.label}
            </span>
            {active === tab.id && (
              <motion.span
                layoutId="tabs-indicator"
                className="absolute inset-x-0 bottom-0 h-0.5 bg-primary"
                transition={{ type: "spring", stiffness: 500, damping: 40 }}
              />
            )}
          </button>
        ))}
      </div>
      <AnimatePresence mode="wait">
        <motion.div
          key={active}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.15, ease: "easeOut" }}
          className="pt-5"
        >
          {children(active)}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
