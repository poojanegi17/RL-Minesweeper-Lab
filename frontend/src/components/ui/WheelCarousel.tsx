import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/cn";

/** State handed to `renderItem` so a card can style its own active treatment
 * without the carousel reaching into its markup. */
export interface WheelItemState {
  index: number;
  isActive: boolean;
}

interface WheelCarouselProps<T> {
  items: T[];
  renderItem: (item: T, state: WheelItemState) => ReactNode;
  getKey: (item: T, index: number) => string;
  /** Card footprint in px. The wheel's radius is derived from this so cards
   * sit tangent to the cylinder without overlapping -- never hardcoded. */
  cardWidth?: number;
  cardHeight?: number;
  ariaLabel?: string;
  className?: string;
  /** Optional detail panel for the active item, floated *over* the wheel.
   *
   * This exists so a card can stay small while its full content still has
   * somewhere to go. Growing the active card in place would force every seat
   * on the cylinder to be sized for the longest item, which makes the whole
   * wheel bigger to serve one card at a time. The panel is rendered outside
   * the stage's `overflow-hidden` so it is free to be larger than a card. */
  renderOverlay?: (item: T, state: WheelItemState) => ReactNode;
}

/** Widest angle between neighbouring cards.
 *
 * The step is normally the true cylinder step, 360/n, so the cards genuinely
 * wrap: the last card sits one seat *behind* the first rather than 176 degrees
 * around the far side, which is what makes selecting across the seam rotate one
 * step instead of most of a turn.
 *
 * The cap only bites at n <= 4, where 360/n would be a half-turn or more
 * (n = 3 gives 120 degrees) and every non-active card would sit behind the
 * viewer. There the cards fan across the front instead, at the cost of an
 * uneven gap where the fan's two ends meet -- an acceptable trade at counts
 * that low, and never reached by the five- and six-card sets on this site. */
const MAX_STEP_DEG = 72;
/** Beyond this the card is edge-on to the viewer and reads as a sliver, so it
 * is faded out and taken out of the hit-testing rather than left as a
 * clickable artifact at the edge of the wheel. Sits above one full step at the
 * cap, so the immediate neighbours on both sides are always visible. */
const VISIBLE_ARC_DEG = 125;
/** Gap between neighbouring cards, as a multiple of the radius at which they
 * would be exactly tangent. At 1.0 the cards touch edge-to-edge, which reads
 * as the wheel being overfull. */
const CARD_GAP_FACTOR = 1.18;
/** Vertical headroom around the cards, as a multiple of card height. The
 * active card scales to 1.04 and casts a soft glow, both of which the
 * container's `overflow-hidden` would otherwise crop. */
const VERTICAL_HEADROOM = 1.16;
/** Breathing room between the outermost card and the container edge. */
const EDGE_MARGIN_PX = 12;

/** Shortest signed rotation from `from` to `to`, so selecting the last card
 * while the first is active spins one step backwards rather than most of a
 * turn forwards. */
function shortestDelta(from: number, to: number): number {
  let delta = (to - from) % 360;
  if (delta > 180) delta -= 360;
  if (delta < -180) delta += 360;
  return delta;
}

function normalizeAngle(deg: number): number {
  let a = deg % 360;
  if (a > 180) a -= 360;
  if (a < -180) a += 360;
  return a;
}

/**
 * A 3D wheel of cards arranged around an invisible vertical cylinder.
 *
 * Each card is placed with `rotateY(i * step) translateZ(radius)` inside a
 * `preserve-3d` wheel, so the depth is real perspective rather than scale
 * faked in 2D: the front card is genuinely nearer the camera and the side
 * cards genuinely curve away from it.
 *
 * The interaction contract is deliberately *sticky*. Hovering a card selects
 * it and the wheel rotates it to the front; moving the mouse away does not
 * put it back. Only selecting a different card changes the active one, which
 * is what makes the wheel explorable by sweeping the mouse across it rather
 * than something that snaps back the moment the pointer leaves.
 *
 * The carousel owns geometry, selection and motion only. It never styles the
 * cards -- `renderItem` receives `isActive` and the card decides what that
 * looks like, so the existing card design is preserved rather than replaced.
 */
export function WheelCarousel<T>({
  items,
  renderItem,
  getKey,
  cardWidth = 300,
  cardHeight = 480,
  ariaLabel = "Card carousel",
  className,
  renderOverlay,
}: WheelCarouselProps<T>) {
  const [activeIndex, setActiveIndex] = useState(0);
  // Tracked unbounded (not derived from activeIndex) so the wheel can keep
  // turning in one direction across the wrap-around instead of unwinding.
  const [rotation, setRotation] = useState(0);
  const [isCompact, setIsCompact] = useState(false);
  // Measured, not assumed: the wheel is shrunk to whatever width it actually
  // gets, so it never widens past its column at any viewport.
  const [stageWidth, setStageWidth] = useState(0);
  // The detail panel is opt-in, not the resting state: the wheel first reads
  // as six plain cards, and only opens once the visitor actually engages with
  // it. Tracked separately from `activeIndex` because index 0 is active from
  // the start -- keying the panel off selection alone would have it open on
  // load, before anyone has touched anything.
  const [expanded, setExpanded] = useState(false);
  const reduceMotion = useReducedMotion();
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => setStageWidth(entry.contentRect.width));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const query = window.matchMedia("(max-width: 767px)");
    const sync = () => setIsCompact(query.matches);
    sync();
    query.addEventListener("change", sync);
    return () => query.removeEventListener("change", sync);
  }, []);

  const count = items.length;
  const step = count > 0 ? Math.min(360 / count, MAX_STEP_DEG) : 0;

  // Cards are chords of the circle: this is the radius at which adjacent cards
  // just touch, plus a little breathing room. Derived from the card width and
  // the step so any item count lays out correctly without tuning.
  const width = isCompact ? Math.min(cardWidth, 260) : cardWidth;
  const height = isCompact ? Math.min(cardHeight, 430) : cardHeight;
  const radius = Math.round((width / 2 / Math.tan((step * Math.PI) / 360)) * CARD_GAP_FACTOR);

  // The widest point of the wheel is the outermost card's far edge. When that
  // exceeds the width this component was actually given, the whole wheel is
  // scaled down with a single CSS transform rather than by recomputing smaller
  // card boxes.
  //
  // That distinction matters: shrinking the boxes numerically leaves the text
  // inside them at its natural size, so a long quote that fits a 320px card
  // overflows a 230px one -- which is exactly what a narrower window produced.
  // A uniform scale takes the type down with the box, so a card is either
  // legible at every width or at none.
  const halfExtent = radius + width / 2;
  const available = stageWidth > 0 ? stageWidth / 2 - EDGE_MARGIN_PX : halfExtent;
  const fit = Math.min(1, Math.max(0.55, available / halfExtent));

  // The stage is taller than the cards so the scaled-up active card and its
  // glow are not clipped; cards are centred inside that headroom.
  const stageHeight = Math.round(height * VERTICAL_HEADROOM * fit);
  const cardTop = Math.round((stageHeight / fit - height) / 2);
  const perspective = isCompact ? 1100 : 1700;

  const select = useCallback(
    (index: number) => {
      if (index < 0 || index >= count) return;
      // Set before the early return: hovering the card that is *already* at
      // the front is still an engagement, and must open it.
      setExpanded(true);
      if (index === activeIndex) return;
      setRotation((current) => current + shortestDelta(current, -index * step));
      setActiveIndex(index);
    },
    [activeIndex, count, step],
  );

  const rotateBy = useCallback(
    (direction: 1 | -1) => select((activeIndex + direction + count) % count),
    [activeIndex, count, select],
  );

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      rotateBy(1);
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      rotateBy(-1);
    }
  };

  if (count === 0) return null;

  const spring = reduceMotion
    ? { duration: 0.2 }
    : { type: "spring" as const, stiffness: 52, damping: 15, mass: 1.05 };

  return (
    <div className={cn("relative flex flex-col items-center gap-8", className)}>
      <div
        ref={containerRef}
        role="group"
        aria-label={ariaLabel}
        tabIndex={0}
        onKeyDown={onKeyDown}
        // `overflow-hidden` is load-bearing, not cosmetic: cards curving round
        // the far side of the cylinder sit hundreds of px off-centre and would
        // widen the page into a horizontal scrollbar on a phone. Clipping here
        // is safe because this element carries `perspective` but not
        // `preserve-3d` -- the flattening rule applies to the wheel inside it,
        // which is untouched.
        className="relative w-full overflow-hidden outline-none focus-visible:ring-2 focus-visible:ring-[#00d2ff]/50"
        style={{ height: stageHeight, perspective: `${perspective}px` }}
      >
        <motion.div
          className="absolute inset-0"
          style={{ transformStyle: "preserve-3d", transformOrigin: "center top" }}
          // Uniform scale on a `preserve-3d` element is safe -- it scales the
          // cards' `translateZ` along with everything else, so the cylinder
          // keeps its proportions instead of flattening.
          animate={{ rotateY: rotation, scale: fit }}
          transition={spring}
          // Touch: a horizontal drag turns the wheel by one card, so the same
          // wheel works where there is no hover to select with.
          drag={isCompact ? "x" : false}
          dragConstraints={{ left: 0, right: 0 }}
          dragElastic={0.12}
          onDragEnd={(_, info) => {
            if (info.offset.x < -40) rotateBy(1);
            else if (info.offset.x > 40) rotateBy(-1);
          }}
        >
          {items.map((item, index) => {
            const relative = normalizeAngle((index - activeIndex) * step);
            const distance = Math.abs(relative) / VISIBLE_ARC_DEG;
            const isActive = index === activeIndex;
            const beyondArc = Math.abs(relative) > VISIBLE_ARC_DEG;

            return (
              <div
                key={getKey(item, index)}
                className="absolute left-1/2"
                style={{
                  width,
                  height,
                  top: cardTop,
                  marginLeft: -width / 2,
                  // The cylinder itself: rotate to this card's seat, then push
                  // out to the surface. Everything else is presentation.
                  transform: `rotateY(${index * step}deg) translateZ(${radius}px)`,
                  transformStyle: "preserve-3d",
                  backfaceVisibility: "hidden",
                  zIndex: Math.round(100 - Math.abs(relative)),
                  pointerEvents: beyondArc ? "none" : "auto",
                }}
              >
                <motion.div
                  className="h-full w-full cursor-pointer"
                  onMouseEnter={() => !isCompact && select(index)}
                  onFocus={() => select(index)}
                  onClick={() => select(index)}
                  animate={{
                    scale: beyondArc ? 0.72 : isActive ? 1.04 : 1 - 0.22 * distance,
                    opacity: beyondArc ? 0 : 1 - 0.42 * distance,
                    filter: `brightness(${beyondArc ? 0.4 : 1 - 0.3 * distance})`,
                  }}
                  transition={spring}
                >
                  {renderItem(item, { index, isActive })}
                </motion.div>
              </div>
            );
          })}
        </motion.div>
      </div>

      {renderOverlay && expanded && (
        // `pointer-events-none` keeps the wheel fully hoverable underneath:
        // the panel is read-only, so it must never become a dead zone that
        // blocks selecting the card it is sitting on top of.
        <div
          className="pointer-events-none absolute inset-x-0 top-0 z-[60] flex items-center justify-center px-4"
          style={{ height: stageHeight }}
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={activeIndex}
              initial={{ opacity: 0, scale: 0.94, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.97, y: -6 }}
              transition={{ duration: 0.28, ease: "easeOut" }}
              // Width is the panel's own business -- this only centres it.
              className="max-w-full"
            >
              {renderOverlay(items[activeIndex], { index: activeIndex, isActive: true })}
            </motion.div>
          </AnimatePresence>
        </div>
      )}

      <div className="flex items-center gap-5">
        <WheelButton label="Previous card" onClick={() => rotateBy(-1)}>
          <ChevronLeft className="h-4 w-4" />
        </WheelButton>

        <div className="flex items-center gap-2">
          {items.map((item, index) => (
            <button
              key={getKey(item, index)}
              type="button"
              onClick={() => select(index)}
              aria-label={`Go to card ${index + 1} of ${count}`}
              aria-current={index === activeIndex}
              className={cn(
                "h-1.5 rounded-full transition-all duration-500",
                index === activeIndex ? "w-7 bg-[#00d2ff]" : "w-1.5 bg-white/25 hover:bg-white/50",
              )}
            />
          ))}
        </div>

        <WheelButton label="Next card" onClick={() => rotateBy(1)}>
          <ChevronRight className="h-4 w-4" />
        </WheelButton>
      </div>
    </div>
  );
}

function WheelButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      className="flex h-9 w-9 items-center justify-center rounded-full border border-white/15 text-white/70 transition-colors hover:border-white/35 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#00d2ff]/50"
    >
      {children}
    </button>
  );
}
