import { createContext, useContext, useEffect, type ReactNode } from "react";

type Theme = "dark";

interface ThemeContextValue {
  theme: Theme;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

/**
 * The whole site is now one fixed dark/cinematic aesthetic (see
 * `Layout.tsx`/`LandingBackground`) -- there's no light mode and no user
 * toggle anymore. This provider still exists only so the several call sites
 * that pick a color variant via `useTheme().theme` (`AGENT_HEX[kind][theme]`
 * in `AgentDetail`/`Compare`/`ResearchPipeline`/etc.) don't each need
 * touching -- it always resolves `"dark"`.
 */
export function ThemeProvider({ children }: { children: ReactNode }) {
  useEffect(() => {
    document.documentElement.classList.add("dark");
  }, []);

  return <ThemeContext.Provider value={{ theme: "dark" }}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}
