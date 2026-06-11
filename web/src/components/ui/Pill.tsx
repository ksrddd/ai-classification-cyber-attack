import { clsx } from "clsx";

type Tone = "ok" | "info" | "warn" | "danger" | "critical" | "muted" | "brand";

export function Pill({
  children,
  tone = "info",
  size = "md",
  className,
}: {
  children: React.ReactNode;
  tone?: Tone;
  size?: "sm" | "md";
  className?: string;
}) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-md font-medium whitespace-nowrap",
        size === "sm" ? "px-1.5 h-5 text-[10.5px]" : "px-2 h-6 text-[11px]",
        tone === "ok"       && "bg-ok/10 text-ok ring-1 ring-ok/25",
        tone === "info"     && "bg-info/10 text-info ring-1 ring-info/25",
        tone === "warn"     && "bg-warn/10 text-warn ring-1 ring-warn/25",
        tone === "danger"   && "bg-danger/10 text-danger ring-1 ring-danger/25",
        tone === "critical" && "bg-critical/10 text-critical ring-1 ring-critical/30",
        tone === "muted"    && "bg-white/[.05] text-ink-1 ring-1 ring-white/[.06]",
        tone === "brand"    && "bg-brand-blue/10 text-brand-cyan ring-1 ring-brand-blue/30",
        className,
      )}
    >
      {children}
    </span>
  );
}
