"use client";
import { useEffect, useRef, useState } from "react";

export function BarRow({
  label,
  value,
  max = 100,
  color = "#3B82F6",
  suffix = "%",
  decimals,
  right,
}: {
  label: string;
  value: number;
  max?: number;
  color?: string;
  suffix?: string;
  /** Decimal places for the value label. Defaults to 2, or 0 when suffix is "". */
  decimals?: number;
  right?: React.ReactNode;
}) {
  const [mounted, setMounted] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setMounted(true);
          obs.disconnect();
        }
      },
      { threshold: 0.1 },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const pct = Math.min(100, (value / max) * 100);
  // Explicit prop wins; fall back to 0 for unit-less values, 2 otherwise
  const dp = decimals ?? (suffix === "" ? 0 : 2);

  return (
    <div ref={ref} className="grid grid-cols-[1fr_auto] gap-3 items-center">
      <div>
        <div className="flex justify-between text-[11.5px] mb-1">
          <span className="text-ink-0">{label}</span>
          <span className="tabular-nums text-ink-1">
            {value.toFixed(dp)}
            {suffix}
          </span>
        </div>
        {/* bg-surface-elevated is theme-aware; replaces dark-only bg-white/[.05] */}
        <div className="h-1 rounded-full bg-surface-elevated overflow-hidden">
          <div
            className="h-full rounded-full transition-[width] duration-700 ease-out"
            style={{ width: mounted ? `${pct}%` : "0%", background: color }}
          />
        </div>
      </div>
      {right}
    </div>
  );
}
