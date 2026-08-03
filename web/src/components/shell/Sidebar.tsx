"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { clsx } from "clsx";
import {
  LayoutDashboard, Database, BarChart2, Trophy, Lightbulb,
  Upload, TrendingUp, Settings, ChevronLeft, ChevronRight, Shield, FileJson,
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

export function Sidebar({
  collapsed,
  onToggle,
}: {
  collapsed: boolean;
  onToggle: () => void;
}) {
  const path = usePathname();

  return (
    <aside
      className={clsx(
        "hidden md:flex flex-col h-full z-30 flex-shrink-0",
        "border-r border-line-base bg-surface",
        "transition-[width] duration-150 ease-out",
        collapsed ? "w-[52px]" : "w-[216px]",
      )}
    >
      {/* Logo */}
      <div className={clsx(
        "flex items-center gap-2.5 h-12 border-b border-line-base flex-shrink-0",
        collapsed ? "px-3.5" : "px-4",
      )}>
        <div className="h-6 w-6 flex-shrink-0 border border-line-strong rounded-sm grid place-items-center">
          <Shield size={13} className="text-info" />
        </div>
        {!collapsed && (
          <div className="min-w-0 flex-1">
            <div className="text-[12.5px] font-semibold text-ink-0 tracking-tight">CyberML</div>
            <div className="text-[9.5px] text-ink-3 font-mono uppercase tracking-[.12em]">NIDS</div>
          </div>
        )}
      </div>

      {/* Which bundle is on screen — a choice, kept visible */}
      <RunSelector collapsed={collapsed} />

      {/* Session */}
      {!collapsed && (
        <div className="px-4 py-2.5 border-b border-line-subtle flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-ok flex-shrink-0" />
          <div className="min-w-0">
            <span className="text-[11px] text-ink-1 font-medium truncate">Sukhum R.</span>
            <span className="text-[10px] text-ink-3 ml-1.5 font-mono">KMITL · 2569</span>
          </div>
        </div>
      )}

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

      {/* Footer */}
      <div className="border-t border-line-base">
        <button className={clsx(
          "w-full flex items-center gap-2.5 h-9 text-[11.5px] text-ink-2 hover:bg-surface-raised hover:text-ink-1 transition-colors duration-100",
          collapsed ? "px-3.5 justify-center" : "px-4",
        )}>
          <Settings size={14} className="flex-shrink-0 text-ink-3" />
          {!collapsed && <span>Settings</span>}
        </button>
        <button
          onClick={onToggle}
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
