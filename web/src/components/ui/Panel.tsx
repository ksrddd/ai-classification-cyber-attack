import { clsx } from "clsx";

export function Panel({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={clsx("rounded bg-surface-raised border border-line-base", className)}>
      {children}
    </div>
  );
}

export function PanelHeader({
  eyebrow,
  title,
  sub,
  right,
}: {
  eyebrow?: string;
  title: React.ReactNode;
  sub?: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-3 px-4 pt-3 pb-2.5 border-b border-line-subtle">
      <div className="min-w-0">
        {eyebrow && (
          <div className="text-[9.5px] uppercase tracking-[.18em] text-ink-3 font-semibold mb-0.5">
            {eyebrow}
          </div>
        )}
        <div className="text-[13px] font-semibold text-ink-0">{title}</div>
        {sub && <div className="text-[11px] text-ink-3 mt-0.5 font-mono">{sub}</div>}
      </div>
      {right && (
        <div className="flex items-center gap-1.5 flex-shrink-0 mt-0.5">{right}</div>
      )}
    </div>
  );
}
