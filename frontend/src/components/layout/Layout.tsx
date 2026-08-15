import { useLayoutEffect, useState, type ReactNode } from "react";
import { useLocation } from "react-router-dom";
import { COLS, LandingBackground } from "@/components/landing/LandingBackground";
import { LandingNavbar } from "@/components/landing/LandingNavbar";
import { LandingFooter } from "@/components/landing/LandingFooter";
import { cn } from "@/lib/cn";

/**
 * Site-wide chrome: the animated Minesweeper background, the sticky glassy
 * navbar, and the footer now wrap every route, not just `/` -- this used to
 * branch between a custom landing experience and the old warm-neutral
 * `Header`/`Footer` shell (still in `components/layout/Header.tsx`, unused
 * now rather than deleted). The landing page (`/`) still renders its own
 * content full-bleed, since `Home.tsx`'s sections (a `min-h-screen` hero,
 * full-width panels) manage their own width and spacing. Every other route
 * gets its existing `max-w-5xl` content wrapped in one glassy liquid-glass
 * panel instead, so pages built from the ordinary `Card`/`Badge`/etc. kit
 * read as "content on a glass card over the board" rather than floating
 * bare over the animated background.
 *
 * `LandingBackground` renders here, as a sibling of `children` (not inside
 * it) deliberately: `children` is `App.tsx`'s page-transition `motion.div`,
 * which animates `y` via a CSS `transform`. A `transform` on any ancestor
 * creates a new containing block for `position: fixed` descendants, which
 * would silently break this component's `fixed inset-0` -- it'd stop
 * tracking the viewport and instead size itself against that motion.div,
 * going effectively invisible. Rendering it outside that subtree avoids the
 * problem entirely rather than relying on Framer Motion never adding a
 * resting transform.
 */
/** Routes whose content panel is a bare container rather than a pane of glass.
 *
 * The panel exists to hold a page's own copy over the animated board. Where a
 * page instead delivers everything through its own cards, the panel is a second
 * sheet of glass behind the first with nothing of its own on it -- the cards
 * end up looking like they are floating on a slab. Opting out here leaves the
 * cards sitting directly on the board, which is exactly how the landing page
 * reads (it renders full-bleed and has no panel at all). */
const BARE_PANEL_ROUTES = new Set(["/about"]);

/** Half-width of the content column the vertical guide lines mark out, in rem. */
const GUIDE_HALF_WIDTH_REM = 36;

/**
 * The x positions of the two vertical guide lines, snapped onto the nearest
 * seam of `LandingBackground`'s grid.
 *
 * These used to be pure CSS at `left-1/2 ± 36rem`. That is a fixed pixel offset,
 * while the board's column pitch is `100vw / COLS` and moves with the window --
 * so the two grids only ever coincided by accident, and in practice never did:
 * measured across 1280/1440/1600/1920 the guides sat 14-23px from the nearest
 * seam every time, and always in opposite directions on the left and right
 * (the guides are symmetric about the centre, the board's seams are not).
 * The result was a line lying mid-tile a few pixels off a real one, which reads
 * as the board itself being irregular.
 *
 * Snapping costs up to half a tile of drift from the true 36rem content edge,
 * which is invisible because nothing else draws that edge -- whereas the board
 * seam is drawn, and being a few pixels off it is not.
 */
function useBoardAlignedGuides(): [number, number] | null {
  const [positions, setPositions] = useState<[number, number] | null>(null);

  // `useLayoutEffect` so the first measurement lands before paint: with a plain
  // effect the lines show up at 0 for a frame and visibly jump into place.
  useLayoutEffect(() => {
    const measure = () => {
      // `clientWidth`, not `innerWidth`: the board is `position: fixed`, so it
      // spans the layout viewport, which excludes a classic scrollbar.
      const viewport = document.documentElement.clientWidth;
      const pitch = viewport / COLS;
      const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
      const half = GUIDE_HALF_WIDTH_REM * rem;
      const snap = (x: number) => Math.round(x / pitch) * pitch;
      setPositions([snap(viewport / 2 - half), snap(viewport / 2 + half)]);
    };

    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  return positions;
}

export function Layout({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  const isLandingPage = pathname === "/";
  const isBarePanel = BARE_PANEL_ROUTES.has(pathname);
  const guides = useBoardAlignedGuides();

  return (
    <div className="min-h-screen">
      <LandingBackground />
      {guides?.map((x, index) => (
        <div
          key={index}
          className="hidden md:block pointer-events-none fixed inset-y-0 w-px bg-white/10 z-[5]"
          style={{ left: x }}
        />
      ))}

      <LandingNavbar />

      {isLandingPage ? (
        children
      ) : (
        <main className="relative z-10 mx-auto w-full max-w-5xl px-4 py-10 sm:px-6">
          {/* `glossy-scope` stays on either branch -- it is what makes the
              cards inside translucent glass rather than solid dark surfaces,
              and that is wanted whether or not the panel itself is glass. */}
          <div
            className={cn(
              "glossy-scope rounded-3xl p-6 sm:p-8",
              !isBarePanel && "liquid-glass border border-white/10",
            )}
          >
            {children}
          </div>
        </main>
      )}

      <LandingFooter />
    </div>
  );
}
