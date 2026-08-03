"use client";

/**
 * Page title, active run, theme.
 *
 * What used to be here: a search field that searched nothing, a notification
 * bell with a permanent unread dot that notified nothing, and a user chip with
 * an avatar and a role badge — on a single-user tool that runs on localhost.
 * All three were borrowed SaaS furniture. A search box that does not search is
 * worse than no search box: it is a promise the interface breaks on first
 * contact, and this interface gets shown to people who are deciding whether to
 * trust it.
 *
 * The right edge now carries the thing that changes meaning on every page —
 * which run is on screen. That matters most exactly when the rail is collapsed
 * or hidden on mobile, where the run selector is not visible at all.
 */

import { Menu } from "lucide-react";
import { ThemeToggle } from "@/components/ui/ThemeToggle";
import { useBundle } from "@/components/bundle/BundleProvider";

export function Topbar({
  title,
  onMobileMenu,
}: {
  title: string;
  onMobileMenu?: () => void;
}) {
  const { active } = useBundle();

  return (
    <header className="h-12 bg-surface border-b border-line-base flex items-center gap-3 px-4 md:px-5 flex-shrink-0">
      <button
        onClick={onMobileMenu}
        aria-label="Open navigation"
        className="md:hidden h-7 w-7 grid place-items-center rounded-sm hover:bg-surface-elevated text-ink-2"
      >
        <Menu size={15} />
      </button>

      <h1 className="text-[12.5px] font-semibold text-ink-0 truncate">{title}</h1>

      <div className="ml-auto flex items-center gap-3 min-w-0">
        {active ? (
          <div className="hidden sm:flex items-baseline gap-2 min-w-0 text-[11px] font-mono">
            <span className="text-ink-3 flex-shrink-0">viewing</span>
            <span className="text-ink-1 truncate" title={`${active.dataset} — ${active.id}`}>
              {active.id}
            </span>
            <span
              className="text-ink-3 flex-shrink-0"
              title={
                active.split_protocol?.includes("temporal")
                  ? "Chronological split"
                  : "Random split — scores are an upper bound"
              }
            >
              {active.split_protocol?.includes("temporal") ? "temporal" : "random"}
            </span>
          </div>
        ) : (
          <span className="hidden sm:block text-[11px] font-mono text-ink-3">
            no run loaded
          </span>
        )}

        <div className="h-4 w-px bg-line-base" aria-hidden />
        <ThemeToggle />
      </div>
    </header>
  );
}
