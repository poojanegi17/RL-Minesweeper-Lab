import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Search } from "lucide-react";
import { LandingLogoMark } from "@/components/landing/LandingPrimitives";

const MENU_ITEMS = ["File", "Edit", "View", "Board", "Agent", "Help"];

const CLOCK_FORMAT = new Intl.DateTimeFormat("en-US", {
  weekday: "short",
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});

/** Purely decorative macOS-menu-bar chrome around the replay viewer
 * centerpiece below -- the "this is a real running application" framing
 * device from the design brief, with a real live clock instead of a
 * hardcoded timestamp. */
export function MenuBarStrip() {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 30_000);
    return () => clearInterval(id);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6, delay: 0.9 }}
      className="relative z-10 h-10 border-t border-b border-white/10 bg-black/40 backdrop-blur-md"
    >
      <div className="mx-auto flex h-full w-full max-w-6xl items-center justify-between px-6 text-xs text-white/70">
        <div className="flex items-center gap-4">
          <LandingLogoMark className="h-4 w-4 rounded" />
          <span className="font-semibold text-white">RL Minesweeper Lab</span>
          {MENU_ITEMS.map((item, i) => (
            <span
              key={item}
              className={i > 3 ? "hidden md:inline" : i > 2 ? "hidden sm:inline" : undefined}
            >
              {item}
            </span>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <Search className="h-3.5 w-3.5" />
          <span className="font-mono">{CLOCK_FORMAT.format(now)}</span>
        </div>
      </div>
    </motion.div>
  );
}
