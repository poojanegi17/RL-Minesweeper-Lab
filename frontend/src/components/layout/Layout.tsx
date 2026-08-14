import type { ReactNode } from "react";
import { useLocation } from "react-router-dom";
import { LandingBackground } from "@/components/landing/LandingBackground";
import { LandingNavbar } from "@/components/landing/LandingNavbar";
import { LandingFooter } from "@/components/landing/LandingFooter";

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
export function Layout({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  const isLandingPage = pathname === "/";

  return (
    <div className="min-h-screen">
      <LandingBackground />
      <div className="hidden md:block pointer-events-none fixed inset-y-0 left-1/2 -translate-x-[calc(50%+36rem)] w-px bg-white/10 z-[5]" />
      <div className="hidden md:block pointer-events-none fixed inset-y-0 left-1/2 translate-x-[calc(-50%+36rem)] w-px bg-white/10 z-[5]" />

      <LandingNavbar />

      {isLandingPage ? (
        children
      ) : (
        <main className="relative z-10 mx-auto w-full max-w-5xl px-4 py-10 sm:px-6">
          <div className="glossy-scope liquid-glass rounded-3xl border border-white/10 p-6 sm:p-8">{children}</div>
        </main>
      )}

      <LandingFooter />
    </div>
  );
}
