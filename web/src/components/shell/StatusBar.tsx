"use client";
import { usePathname } from "next/navigation";

export function StatusBar() {
  const path = usePathname();
  const routeName = path === "/" ? "overview" : path.slice(1).split("/")[0];

  return (
    <div className="bg-surface border-t border-line-base flex-shrink-0">
      <div className="px-4 md:px-5 h-7 flex items-center gap-3 text-[10px] text-ink-3 font-mono">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-sm bg-ok" />
          <span>API</span>
        </span>
        <span className="text-line-strong">|</span>
        <span className="hidden md:inline tabular-nums">78 features · 7 classes · CICIDS2017</span>
        <span className="ml-auto flex items-center gap-3">
          <span className="hidden md:inline">/{routeName}</span>
          <span>v3.2.1</span>
        </span>
      </div>
    </div>
  );
}
