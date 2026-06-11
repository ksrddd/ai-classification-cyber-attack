"use client";
import { clsx } from "clsx";
import { Sparkline } from "./Sparkline";

export function KpiCard({
  label,
  value,
  sub,
  color = "#A8AFC0",
  className,
  spark,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  color?: string;
  className?: string;
  spark?: number[];
}) {
  return (
    <div
      className={clsx(
        "relative bg-surface-raised border border-line-base rounded p-4 overflow-hidden",
        "hover:bg-surface-elevated transition-colors duration-100",
        className,
      )}
      style={{ borderLeftColor: color, borderLeftWidth: "2px" }}
    >
      {spark && (
        <div className="absolute right-0 top-0 bottom-0 w-2/5 opacity-15 pointer-events-none">
          <Sparkline data={spark} color={color} h={72} w={160} />
        </div>
      )}
      <div className="relative text-[10px] uppercase tracking-[.16em] text-ink-3 font-medium mb-2">
        {label}
      </div>
      <div className="relative tabular-nums text-[24px] leading-none font-semibold font-mono tracking-tight text-ink-0">
        {value}
      </div>
      {sub && (
        <div className="relative text-[10.5px] text-ink-3 mt-2 font-mono">{sub}</div>
      )}
    </div>
  );
}
