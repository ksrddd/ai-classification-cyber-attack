const SEV_MAP: Record<string, [string, string]> = {
  critical: ["#F43F5E", "CRITICAL"],
  danger:   ["#EF4444", "HIGH"],
  warn:     ["#F59E0B", "MED"],
  info:     ["#38BDF8", "INFO"],
  ok:       ["#10B981", "OK"],
};

export function Sev({ sev }: { sev: string }) {
  const [color, label] = SEV_MAP[sev] ?? SEV_MAP.info;
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="h-1.5 w-1.5 rounded-full flex-shrink-0"
        style={{ background: color, boxShadow: `0 0 8px ${color}` }}
      />
      <span
        className="text-[10.5px] font-semibold tracking-wider"
        style={{ color }}
      >
        {label}
      </span>
    </span>
  );
}
