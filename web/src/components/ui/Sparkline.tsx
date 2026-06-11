"use client";
import { useId } from "react";

function smoothPath(pts: { x: number; y: number }[]): string {
  if (pts.length < 2) return "";
  const p = (i: number) => pts[Math.max(0, Math.min(pts.length - 1, i))];
  const d: string[] = [`M ${p(0).x.toFixed(2)} ${p(0).y.toFixed(2)}`];
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = p(i - 1), p1 = p(i), p2 = p(i + 1), p3 = p(i + 2);
    const cp1x = p1.x + (p2.x - p0.x) / 6, cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6, cp2y = p2.y - (p3.y - p1.y) / 6;
    d.push(`C ${cp1x.toFixed(2)} ${cp1y.toFixed(2)}, ${cp2x.toFixed(2)} ${cp2y.toFixed(2)}, ${p2.x.toFixed(2)} ${p2.y.toFixed(2)}`);
  }
  return d.join(" ");
}

export function Sparkline({
  data,
  color = "#3B82F6",
  h = 32,
  w = 120,
  area = true,
}: {
  data: number[];
  color?: string;
  h?: number;
  w?: number;
  area?: boolean;
}) {
  const id = useId().replace(/:/g, "s");

  if (data.length < 2) return null;
  const max = Math.max(...data), min = Math.min(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => ({
    x: 1 + (i * (w - 2)) / (data.length - 1),
    y: h - 2 - ((v - min) / range) * (h - 4),
  }));
  const line = smoothPath(pts);
  const areaPath = `${line} L ${pts[pts.length - 1].x.toFixed(2)} ${h} L ${pts[0].x.toFixed(2)} ${h} Z`;

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="w-full h-full"
      preserveAspectRatio="none"
      aria-hidden
    >
      {area && (
        <>
          <defs>
            <linearGradient id={id} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.4" />
              <stop offset="100%" stopColor={color} stopOpacity="0" />
            </linearGradient>
          </defs>
          <path d={areaPath} fill={`url(#${id})`} />
        </>
      )}
      <path
        d={line}
        stroke={color}
        strokeWidth="1.5"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
