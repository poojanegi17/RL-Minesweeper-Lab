import { NavLink } from "react-router-dom";
import { motion } from "framer-motion";
import { Bomb, Moon, Sun } from "lucide-react";
import { useTheme } from "@/app/ThemeProvider";
import { cn } from "@/lib/cn";

const NAV_LINKS = [
  { to: "/", label: "Home" },
  { to: "/agents", label: "Agents" },
  { to: "/research", label: "Research" },
  { to: "/compare", label: "Compare" },
  { to: "/replay", label: "Replay" },
  { to: "/about", label: "About" },
];

export function Header() {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex w-full max-w-5xl items-center justify-between px-4 py-3.5 sm:px-6">
        <NavLink to="/" className="flex items-center gap-2.5">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Bomb className="h-4 w-4" />
          </span>
          <span className="font-semibold tracking-tight text-heading">
            RL Minesweeper Lab
          </span>
        </NavLink>

        <nav className="flex items-center gap-1">
          {NAV_LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === "/"}
              className="relative px-3 py-1.5 text-sm font-medium"
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <motion.span
                      layoutId="nav-indicator"
                      className="absolute inset-0 rounded-lg bg-surface"
                      transition={{ type: "spring", stiffness: 500, damping: 40 }}
                    />
                  )}
                  <span
                    className={cn(
                      "relative z-10 transition-colors",
                      isActive
                        ? "text-heading"
                        : "text-text-muted hover:text-text",
                    )}
                  >
                    {link.label}
                  </span>
                </>
              )}
            </NavLink>
          ))}
          <button
            type="button"
            onClick={toggleTheme}
            aria-label="Toggle color theme"
            className="ml-1.5 flex h-8 w-8 items-center justify-center rounded-lg text-text-muted transition-colors hover:bg-surface hover:text-text"
          >
            {theme === "dark" ? (
              <Sun className="h-4 w-4" />
            ) : (
              <Moon className="h-4 w-4" />
            )}
          </button>
        </nav>
      </div>
    </header>
  );
}
