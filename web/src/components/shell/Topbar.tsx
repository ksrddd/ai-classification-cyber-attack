"use client";

import { Bell, Menu, Search } from "lucide-react";
import { ThemeToggle } from "@/components/ui/ThemeToggle";

export function Topbar({
  title,
  onMobileMenu,
}: {
  title: string;
  onMobileMenu?: () => void;
}) {
  return (
    <div className="h-12 bg-surface border-b border-line-base flex items-center gap-3 px-4 md:px-5 flex-shrink-0">
      <button
        onClick={onMobileMenu}
        className="md:hidden h-7 w-7 grid place-items-center rounded-sm hover:bg-surface-elevated text-ink-2"
      >
        <Menu size={15} />
      </button>

      {/* Breadcrumb */}
      <div className="hidden md:flex items-center gap-1.5 text-[11.5px] text-ink-3 min-w-0 font-mono">
        <span>cyberml</span>
        <span>/</span>
        <span className="text-ink-1 font-semibold truncate">{title.toLowerCase()}</span>
      </div>

      {/* Search */}
      <div className="ml-auto hidden sm:block w-full max-w-xs">
        <div className="relative">
          <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-3 pointer-events-none" />
          <input
            className="w-full h-7 rounded-sm bg-surface-raised border border-line-base hover:border-line-strong focus:border-info/50 transition-colors pl-7 pr-12 text-[11.5px] text-ink-1 placeholder:text-ink-3 outline-none font-mono"
            placeholder="search metrics, features…"
          />
          <span className="absolute right-2 top-1/2 -translate-y-1/2 flex gap-0.5 pointer-events-none">
            <kbd className="font-mono text-[9.5px] px-1 py-0.5 rounded-sm bg-surface-elevated border border-line-base text-ink-3">⌘K</kbd>
          </span>
        </div>
      </div>

      <div className="flex items-center gap-1 flex-shrink-0">
        {/* Theme toggle */}
        <ThemeToggle />

        <div className="h-4 w-px bg-line-base mx-0.5" />

        {/* Notification */}
        <button className="relative h-7 w-7 grid place-items-center rounded-sm hover:bg-surface-elevated transition-colors">
          <Bell size={14} className="text-ink-2" />
          <span className="absolute top-1.5 right-1.5 h-1.5 w-1.5 rounded-full bg-critical" />
        </button>

        <div className="h-4 w-px bg-line-base mx-1" />

        {/* User */}
        <div className="flex items-center gap-2">
          <div className="h-6 w-6 rounded-sm bg-surface-elevated border border-line-base grid place-items-center text-[9.5px] font-semibold font-mono text-ink-1">
            SR
          </div>
          <div className="hidden md:block leading-tight">
            <div className="text-[11.5px] font-medium text-ink-0">Sukhum R.</div>
            <div className="text-[9.5px] text-ink-3 font-mono">L4 · Senior Project</div>
          </div>
        </div>
      </div>
    </div>
  );
}
