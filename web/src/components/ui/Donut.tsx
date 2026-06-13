export function Donut({
  data,
  size = 180,
  thickness = 16,
}: {
  data: { pct: number; color: string; name?: string }[];
  size?: number;
  thickness?: number;
}) {
  const R = size / 2 - thickness / 2 - 2;
  const C = 2 * Math.PI * R;
  let acc = 0;

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      className="w-full h-full"
      style={{ transform: "rotate(-90deg)" }}
      aria-hidden
    >
      <circle
        cx={size / 2}
        cy={size / 2}
        r={R}
        fill="none"
        stroke="rgba(255,255,255,.04)"
        strokeWidth={thickness}
      />
      {data.map((d, i) => {
        const len = (d.pct / 100) * C;
        const off = (acc / 100) * C;
        acc += d.pct;
        return (
          <circle
            key={d.name ?? i}  /* stable key — name is unique per class */
            cx={size / 2}
            cy={size / 2}
            r={R}
            fill="none"
            stroke={d.color}
            strokeWidth={thickness}
            strokeLinecap="butt"
            strokeDasharray={`${len.toFixed(2)} ${(C - len).toFixed(2)}`}
            strokeDashoffset={(-off).toFixed(2)}
          />
        );
      })}
    </svg>
  );
}
