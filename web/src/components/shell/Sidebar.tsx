"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef } from "react";
import { clsx } from "clsx";
import {
  LayoutDashboard, Database, BarChart2, Trophy, Lightbulb,
  Upload, TrendingUp, ChevronLeft, ChevronRight, Shield, FileJson,
  X,
} from "lucide-react";
import { RunSelector } from "@/components/bundle/RunSelector";

// Grouped by what the user is doing, per the redesign mockup: look at the
// data, then at the results, then use the model.
const NAV = [
  { href: "/",            label: "Overview",    icon: LayoutDashboard, group: "Data"    },
  { href: "/dataset",     label: "Dataset",     icon: Database,        group: "Data"    },
  { href: "/eda",         label: "Distributions", icon: TrendingUp,    group: "Data"    },
  { href: "/performance", label: "Model detail", icon: BarChart2,      group: "Results" },
  { href: "/compare",     label: "Comparison",  icon: Trophy,          group: "Results" },
  { href: "/shap",        label: "Explainability", icon: Lightbulb,    group: "Results" },
  { href: "/predict",     label: "Batch predict", icon: Upload,        group: "Use"     },
  { href: "/contract",    label: "Bundle contract", icon: FileJson,    group: "Use"     },
];

const GROUPS = ["Data", "Results", "Use"];

/**
 * Nav body, shared by the desktop rail and the mobile drawer.
 *
 * Extracted so the two never drift. A drawer that lists a different set of
 * destinations than the rail is a bug that only ever shows up on a phone,
 * which is exactly where nobody looks.
 */
function NavBody({
  collapsed,
  onNavigate,
}: {
  collapsed: boolean;
  onNavigate?: () => void;
}) {
  const path = usePathname();

  return (
    <>
      {/* What the tool is, stated once.
          The two-line brand lockup that was here (icon tile + "CyberML" +
          "NIDS") was product furniture for a product that does not exist:
          this runs on localhost for one team. A description outlives a
          made-up name, and the rail below already says which corpus is
          loaded. The green "online" dot and the name badge went with it —
          there is no session and no one else to be. */}
      <div
        className={clsx(
          "flex items-center h-12 border-b border-line-base flex-shrink-0",
          collapsed ? "px-3.5 justify-center" : "px-4",
        )}
      >
        {collapsed ? (
          <Shield size={14} className="text-ink-2" aria-hidden />
        ) : (
          <span className="text-[11.5px] text-ink-1 leading-tight">
            NIDS model comparison
          </span>
        )}
      </div>

      {/* Which bundle is on screen — a choice, kept visible */}
      <RunSelector collapsed={collapsed} />

      {/* Nav */}
      <nav className="flex-1 flex flex-col py-2 overflow-y-auto">
        {GROUPS.map((group) => {
          const items = NAV.filter((n) => n.group === group);
          return (
            <div key={group} className="mb-1">
              {!collapsed && (
                <div className="text-[9px] uppercase tracking-[.2em] text-ink-3 px-4 py-1.5 font-semibold">
                  {group}
                </div>
              )}
              {items.map((n) => {
                const isActive = n.href === "/" ? path === "/" : path.startsWith(n.href);
                const Icon = n.icon;
                return (
                  <Link
                    key={n.href}
                    href={n.href}
                    onClick={onNavigate}
                    aria-current={isActive ? "page" : undefined}
                    className={clsx(
                      "flex items-center gap-2.5 h-8 text-[12px] font-medium transition-colors duration-100",
                      collapsed ? "px-3.5 justify-center" : "px-4",
                      isActive
                        ? "bg-surface-elevated text-ink-0 border-l-2 border-info"
                        : "text-ink-2 hover:bg-surface-raised hover:text-ink-1 border-l-2 border-transparent",
                    )}
                  >
                    <Icon
                      size={14}
                      className={clsx(
                        "flex-shrink-0",
                        isActive ? "text-info" : "text-ink-3",
                      )}
                    />
                    {!collapsed && <span className="truncate">{n.label}</span>}
                  </Link>
                );
              })}
            </div>
          );
        })}
      </nav>
    </>
  );
}

export function Sidebar({
  collapsed,
  onToggle,
}: {
  collapsed: boolean;
  onToggle: () => void;
}) {
  return (
    <aside
      className={clsx(
        "hidden md:flex flex-col h-full z-30 flex-shrink-0",
        "border-r border-line-base bg-surface",
        "transition-[width] duration-150 ease-out",
        collapsed ? "w-[52px]" : "w-[216px]",
      )}
    >
      <NavBody collapsed={collapsed} />

      {/* Footer.
          A "Settings" button used to sit here with no settings behind it.
          Every preference this tool actually has — theme, active run — is
          already a control somewhere it belongs. */}
      <div className="border-t border-line-base">
        <button
          onClick={onToggle}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-expanded={!collapsed}
          className={clsx(
            "w-full flex items-center gap-2 h-8 text-[11px] text-ink-3 hover:text-ink-1 hover:bg-surface-raised transition-colors duration-100",
            collapsed ? "px-3.5 justify-center" : "px-4",
          )}
        >
          {collapsed ? <ChevronRight size={13} /> : <><ChevronLeft size={13} /><span>Collapse</span></>}
        </button>
      </div>
    </aside>
  );
}

/**
 * Mobile navigation drawer.
 *
 * The rail is `hidden md:flex`, and the Topbar's hamburger was wired to
 * nothing — below `md` this application had no navigation whatsoever. This is
 * that missing half: a real drawer with a scrim, Escape to close, and focus
 * moved into the panel on open so a keyboard user is not left behind on the
 * page underneath.
 */
export function MobileNav({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    panel.current?.focus();
    // The scrim covers the page; letting it scroll underneath is disorienting.
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="md:hidden fixed inset-0 z-40">
      <button
        aria-label="Close navigation"
        onClick={onClose}
        className="absolute inset-0 h-full w-full bg-black/60 animate-fadeIn"
      />
      <div
        ref={panel}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label="Navigation"
        className="absolute inset-y-0 left-0 w-[248px] max-w-[82vw] flex flex-col bg-surface border-r border-line-base outline-none"
      >
        <NavBody collapsed={false} onNavigate={onClose} />
        <div className="border-t border-line-base">
          <button
            onClick={onClose}
            className="w-full flex items-center gap-2 h-9 px-4 text-[11px] text-ink-2 hover:text-ink-0 hover:bg-surface-raised transition-colors duration-100"
          >
            <X size={13} aria-hidden />
            <span>Close</span>
          </button>
        </div>
      </div>
    </div>
  );
}
