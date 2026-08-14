import { Children, isValidElement, useMemo, type ReactNode } from "react";
import { Dropdown, type DropdownOption } from "@/components/ui/Dropdown";

interface SelectProps {
  value?: string;
  /** Shaped like the part of a change event the call sites actually read, so
   * the `(e) => setX(e.target.value)` handlers written against the native
   * element keep working unchanged. */
  onChange?: (event: { target: { value: string } }) => void;
  className?: string;
  children?: ReactNode;
  "aria-label"?: string;
}

/** Reads `<option>` children into the `{value, label}` list `Dropdown` takes,
 * so callers keep writing familiar `<Select><option value=…>` markup. */
function readOptions(children: ReactNode): DropdownOption[] {
  const options: DropdownOption[] = [];
  Children.forEach(children, (child) => {
    if (!isValidElement(child)) return;
    const props = child.props as { value?: string | number; children?: ReactNode };
    const value = props.value != null ? String(props.value) : "";
    options.push({
      value,
      label: typeof props.children === "string" ? props.children : String(props.children ?? value),
    });
  });
  return options;
}

/**
 * `<Select>` used to render a real `<select>`, whose *open* popup list the
 * browser paints outside the normal pipeline -- so the dark theme reached the
 * closed field and stopped there, leaving a white menu over a dark page.
 * `color-scheme: dark` asks engines to render that popup dark but cannot
 * compel it, and on some platforms the list is an OS widget that ignores it
 * entirely.
 *
 * `Dropdown` already solved this properly for the board/level pickers, which
 * is exactly why those looked right while these did not. So this is now a thin
 * adapter onto it rather than a second implementation: same popup, same
 * tokens, same animation, one place to change. The `<option>`-children API is
 * kept so no call site had to move.
 */
export function Select({ value, onChange, className, children, ...props }: SelectProps) {
  const options = useMemo(() => readOptions(children), [children]);

  return (
    <Dropdown
      value={value ?? ""}
      options={options}
      onChange={(next) => onChange?.({ target: { value: next } })}
      ariaLabel={props["aria-label"] ?? ""}
      className={className}
    />
  );
}
