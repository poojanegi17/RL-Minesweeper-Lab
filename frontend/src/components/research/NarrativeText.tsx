/**
 * Renders a narrative field that may be either a single sentence or a list of
 * short points.
 *
 * `VARIANT_NARRATIVES`' `limitation`/`conclusion` entries are authored as
 * arrays of plain-language points, since the findings behind them usually have
 * several independent parts (a number, a mechanism, a consequence) that read
 * badly welded into one paragraph. `ResearchPipeline`'s `STEPS` still authors
 * single strings, so both shapes are supported rather than forcing one.
 */
export type Narrative = string | readonly string[];

interface NarrativeTextProps {
  value: Narrative;
  /** Extra classes for the text itself, so callers keep control of colour/size. */
  className?: string;
}

export function NarrativeText({ value, className }: NarrativeTextProps) {
  if (typeof value === "string") {
    return <p className={className}>{value}</p>;
  }

  return (
    <ul className={`flex flex-col gap-1.5 ${className ?? ""}`}>
      {value.map((point) => (
        <li key={point} className="flex gap-2">
          <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-current opacity-50" aria-hidden="true" />
          <span>{point}</span>
        </li>
      ))}
    </ul>
  );
}
