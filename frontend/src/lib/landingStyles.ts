/** The shiny-gradient text treatment used on the hero headline word and the
 * board-configurations watermark -- same gradient stops both places, just a
 * different noise filter id (see `NoiseFilters`) for the intensity each
 * context calls for. Kept in its own module (not alongside the landing
 * components) purely so those files can stay component-only for fast
 * refresh. */
export const shinyGradientStyle: React.CSSProperties = {
  backgroundImage:
    "linear-gradient(to right, #091020 0%, #0B2551 12.5%, #A4F4FD 32.5%, #00d2ff 50%, #0B2551 67.5%, #091020 87.5%, #091020 100%)",
  backgroundSize: "200% auto",
  WebkitBackgroundClip: "text",
  backgroundClip: "text",
  color: "transparent",
  WebkitTextFillColor: "transparent",
  filter: "url(#landing-noise-subtle)",
};

/** The same hue family as `shinyGradientStyle`, held still.
 *
 * The animated version sweeps a 200%-wide gradient across the text forever,
 * which on a page where the content is meant to be the focal point reads as
 * decoration competing with it. This one spans the gradient once across the
 * headline and stops, keeping the brand colour without the motion -- and
 * without the noise filter, which softened the glyph edges at display sizes.
 */
export const staticHeadlineGradient: React.CSSProperties = {
  backgroundImage: "linear-gradient(102deg, #A4F4FD 0%, #00d2ff 42%, #3d81e3 100%)",
  WebkitBackgroundClip: "text",
  backgroundClip: "text",
  color: "transparent",
  WebkitTextFillColor: "transparent",
};

/** Small tracked mono line that sits above a headline. JetBrains Mono is the
 * face this project already uses for every measured figure, so a mono kicker
 * reads as "this is a lab page" rather than as another display font. */
export const headlineEyebrowClass =
  "mb-4 font-mono text-[11px] font-medium tracking-[0.22em] text-white/40 uppercase";
