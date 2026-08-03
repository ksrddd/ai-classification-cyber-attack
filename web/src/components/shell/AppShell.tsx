"use client";

import { useState } from "react";
import { MobileNav, Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";
import { StatusBar } from "./StatusBar";

export function AppShell({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  const [collapsed, setCollapsed] = useState(false);
  // The rail is desktop-only. Below `md` this drawer is the only navigation
  // there is — the Topbar hamburger was previously wired to nothing.
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden bg-canvas">
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((c) => !c)} />
      <MobileNav open={mobileOpen} onClose={() => setMobileOpen(false)} />

      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <Topbar title={title} onMobileMenu={() => setMobileOpen(true)} />
        <main className="flex-1 overflow-y-auto px-4 md:px-6 py-5">
          {children}
        </main>
        <StatusBar />
      </div>
    </div>
  );
}
