"use client";
import { useEffect, useRef, useState } from "react";

export function BarRow({
  label,
  value,
  max = 100,
  color = "#3B82F6",
  suffix = "%",
  right,
}: {
  label: string;
  value: number;
  max?: number;
  color?: string;
  suffix?: string;
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

  return (
    <div ref={ref} className="grid grid-cols-[1fr_auto] gap-3 items-center">
      <div>
        <div className="flex justify-between text-[11.5px] mb-1">
          <span className="text-ink-0">{label}</span>
          <span className="tabular-nums text-ink-1">
            {value.toFixed(suffix === "" ? 0 : 2)}
            {suffix}
          </span>
        </div>
        <div className="h-1 rounded-full bg-white/[.05] overflow-hidden">
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
