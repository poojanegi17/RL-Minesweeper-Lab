/**
 * Two SVG grain filters shared across the landing page's shiny-text
 * treatments: a subtle multiply-blend grain for the hero headline word, and a
 * stronger overlay-blend grain for the board-configurations watermark text.
 * Mounted once at the page root (`Home.tsx`) and referenced elsewhere via
 * `filter: url(#landing-noise-subtle)` / `url(#landing-noise-watermark)` --
 * distinct ids (unlike the original design brief's single reused id) since
 * two elements sharing one SVG id is invalid and only the first would ever
 * resolve.
 */
export function NoiseFilters() {
  return (
    <svg width="0" height="0" className="absolute" aria-hidden="true">
      <defs>
        <filter id="landing-noise-subtle">
          <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves={2} stitchTiles="stitch" />
          <feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.35 0" />
          <feComposite in2="SourceGraphic" operator="in" result="noise" />
          <feBlend in="SourceGraphic" in2="noise" mode="multiply" />
        </filter>
        <filter id="landing-noise-watermark">
          <feTurbulence type="fractalNoise" baseFrequency="0.5" numOctaves={2} stitchTiles="stitch" />
          <feComponentTransfer>
            <feFuncA type="linear" slope={0.075} />
          </feComponentTransfer>
          <feComposite in2="SourceGraphic" operator="in" result="noise" />
          <feBlend in="SourceGraphic" in2="noise" mode="overlay" />
        </filter>
      </defs>
    </svg>
  );
}
