import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import { motion } from "framer-motion";
import { Menu, X } from "lucide-react";
import { LandingLogoMark, GlassPillButton } from "@/components/landing/LandingPrimitives";
import { cn } from "@/lib/cn";

const NAV_LINKS = [
  { to: "/", label: "Home", end: true },
  { to: "/agents", label: "Agents" },
  { to: "/research", label: "Research" },
  { to: "/compare", label: "Compare" },
  { to: "/replay", label: "Replay" },
  { to: "/about", label: "About" },
];

const SCROLL_THRESHOLD = 24;

/** Fully transparent over the hero, then smoothly gains a glassy dark
 * backing once scrolled -- readability against whatever section happens to
 * be behind it -- while staying `sticky` so it never scrolls away. */
export function LandingNavbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > SCROLL_THRESHOLD);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <motion.nav
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className={cn(
        "sticky top-0 z-30 transition-colors duration-300",
        scrolled ? "border-b border-white/10 bg-[#0b0c0e]/70 backdrop-blur-md" : "border-b border-transparent bg-transparent",
      )}
    >
      <div className="relative mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-6">
        <NavLink to="/" className="flex items-center gap-2.5" end>
          <LandingLogoMark />
          <span className="font-semibold tracking-tight text-white">RL Minesweeper Lab</span>
        </NavLink>

        <div className="hidden items-center gap-1 md:flex">
          {NAV_LINKS.map((link, i) => (
            <motion.div
              key={link.to}
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 + i * 0.05, ease: "easeOut" }}
            >
              <NavLink to={link.to} end={link.end} className="relative px-3 py-1.5 text-sm font-medium">
                {({ isActive }) => (
                  <>
                    {isActive && (
                      <motion.span
                        layoutId="landing-nav-indicator"
                        className="absolute inset-0 rounded-full border border-white/10 bg-black/50 shadow-[inset_0_1px_0_rgba(255,255,255,0.08)] backdrop-blur-sm"
                        transition={{ type: "spring", stiffness: 500, damping: 40 }}
                      />
                    )}
                    <span
                      className={cn(
                        "relative z-10 transition-colors",
                        isActive ? "text-white" : "text-white/70 hover:text-white",
                      )}
                    >
                      {link.label}
                    </span>
                  </>
                )}
              </NavLink>
            </motion.div>
          ))}
        </div>

        <div className="hidden md:flex">
          <GlassPillButton label="Explore Agents" to="/agents" />
        </div>

        <button
          type="button"
          onClick={() => setMobileOpen((v) => !v)}
          aria-label="Toggle menu"
          aria-expanded={mobileOpen}
          className="flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-white/5 text-white md:hidden"
        >
          {mobileOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        </button>

        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            className="liquid-glass absolute inset-x-6 top-full z-30 mt-2 flex flex-col gap-4 rounded-2xl p-5 md:hidden"
          >
            {NAV_LINKS.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.end}
                onClick={() => setMobileOpen(false)}
                className={({ isActive }) =>
                  cn("text-sm font-medium", isActive ? "text-white" : "text-white/70")
                }
              >
                {link.label}
              </NavLink>
            ))}
            <GlassPillButton label="Explore Agents" to="/agents" full />
          </motion.div>
        )}
      </div>
    </motion.nav>
  );
}
