import { classColor } from "@/lib/colors";

export function ClassChip({ name, size = "sm" }: { name: string; size?: "sm" | "md" }) {
  const color = classColor(name);
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded font-medium ${
        size === "sm" ? "text-[11px]" : "text-[12.5px]"
      }`}
    >
      <span className="h-2 w-2 rounded-sm flex-shrink-0" style={{ background: color }} />
      <span className="text-ink-0">{name}</span>
    </span>
  );
}
