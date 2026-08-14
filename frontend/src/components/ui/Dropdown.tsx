import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "@/lib/cn";

export interface DropdownOption {
  value: string;
  label: string;
}

interface DropdownProps {
  value: string;
  options: DropdownOption[];
  onChange: (value: string) => void;
  ariaLabel: string;
  className?: string;
}

/**
 * A fully custom dropdown -- not a native `<select>`. A native select's
 * *closed* field can be themed with ordinary CSS, but its *open* popup list
 * is rendered by the browser/OS outside the normal paint pipeline in
 * virtually every browser: `background-color`, `backdrop-filter`, and this
 * app's dark theme simply don't reach it, so it always shows the platform's
 * own (usually light) menu chrome no matter what the closed field looks
 * like. Building the popup out of ordinary positioned `div`s instead is the
 * only way to make an open dropdown actually match the page around it.
 *
 * Uses the same `bg-surface`/`border-border` tokens `Select`/`Card` do, so
 * it stays dark automatically (site-wide now, see `ThemeProvider`) and goes
 * glassy automatically wherever an ancestor opts into `.glossy-scope`,
 * rather than a hardcoded one-off color scheme.
 */
export function Dropdown({ value, options, onChange, ariaLabel, className }: DropdownProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const selected = options.find((o) => o.value === value);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: PointerEvent) {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) setOpen(false);
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div ref={rootRef} className={cn("relative", className)}>
      <button
        type="button"
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 rounded-lg border border-border bg-surface py-2 pr-3 pl-3 text-sm text-text transition-colors hover:border-text-muted/50 focus:border-primary focus:outline-none"
      >
        <span className="truncate">{selected?.label ?? ariaLabel}</span>
        <ChevronDown className={cn("h-3.5 w-3.5 shrink-0 text-text-muted transition-transform", open && "rotate-180")} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.ul
            role="listbox"
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
            className="absolute top-full left-0 z-30 mt-1.5 w-full min-w-max overflow-hidden rounded-lg border border-border bg-surface shadow-lg shadow-black/20"
          >
            {options.map((option) => {
              const isSelected = option.value === value;
              return (
                <li key={option.value} role="option" aria-selected={isSelected}>
                  <button
                    type="button"
                    onClick={() => {
                      onChange(option.value);
                      setOpen(false);
                    }}
                    className={cn(
                      "flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-sm transition-colors",
                      isSelected ? "bg-surface-hover text-heading" : "text-text hover:bg-surface-hover",
                    )}
                  >
                    <span className="truncate">{option.label}</span>
                    {isSelected && <Check className="h-3.5 w-3.5 shrink-0 text-primary" />}
                  </button>
                </li>
              );
            })}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}
