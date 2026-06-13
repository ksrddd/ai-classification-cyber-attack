"use client";

import { useEffect, useState } from "react";
import { Sun, Moon } from "lucide-react";
import { clsx } from "clsx";

type Theme = "dark" | "light";

function applyTheme(theme: Theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem("theme", theme); } catch (_) {}
}

export function ThemeToggle({ className }: { className?: string }) {
  const [theme, setTheme] = useState<Theme>("dark");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const stored = localStorage.getItem("theme") as Theme | null;
    if (stored === "light" || stored === "dark") {
      setTheme(stored);
    } else {
      setTheme(window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    }
  }, []);

  const toggle = () => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    applyTheme(next);
  };

  /* Avoid hydration mismatch — render placeholder until mounted */
  if (!mounted) {
    return (
      <span
        className={clsx(
          "h-7 w-7 rounded-sm bg-surface-raised",
          className,
        )}
      />
    );
  }

  return (
    <button
      onClick={toggle}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
      className={clsx(
        "h-7 w-7 rounded-sm grid place-items-center",
        "text-ink-2 hover:text-ink-0 hover:bg-surface-raised",
        "focus-visible:outline-none focus-visible:ring-2",
        "focus-visible:ring-[rgb(var(--border-focus))]",
        "transition-colors duration-100",
        className,
      )}
    >
      {theme === "dark"
        ? <Sun size={14} strokeWidth={1.75} />
        : <Moon size={14} strokeWidth={1.75} />}
    </button>
  );
}
