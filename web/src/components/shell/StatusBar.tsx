"use client";

/**
 * Connection state and the shape of the loaded run.
 *
 * Every value here used to be a constant: a green dot hardcoded to "ok", the
 * literal string "77 features · 9 classes · CICIDS2017", and an invented
 * version number. It reported a healthy API and a loaded 2017 corpus while the
 * application was serving 404s for its own JavaScript, and it kept saying
 * "CICIDS2017 · 9 classes" while a 15-class 2018 run was on screen.
 *
 * A status bar that cannot be wrong is not a status bar. Each field now
 * reflects something real, or says nothing.
 */

import { clsx } from "clsx";
import { useBundle } from "@/components/bundle/BundleProvider";
import { count } from "@/lib/bundles";

export function StatusBar() {
  const { active, loading, error } = useBundle();

  const state = error
    ? { tone: "bg-danger", label: "API unreachable" }
    : loading
      ? { tone: "bg-ink-3", label: "connecting" }
      : { tone: "bg-ok", label: "API" };

  return (
    <footer className="bg-surface border-t border-line-base flex-shrink-0">
      <div className="px-4 md:px-5 h-7 flex items-center gap-3 text-[10px] text-ink-3 font-mono">
        <span className="inline-flex items-center gap-1.5">
          <span
            className={clsx("h-1.5 w-1.5 rounded-sm", state.tone)}
            aria-hidden
          />
          <span className={error ? "text-danger" : undefined}>{state.label}</span>
        </span>

        {active && (
          <>
            <span className="text-line-strong" aria-hidden>
              |
            </span>
            <span className="hidden md:inline tabular-nums truncate">
              {active.n_features ?? "?"} features · {active.n_classes} classes ·{" "}
              {count(active.n_test)} test flows
            </span>
          </>
        )}

        {active?.hp_tuned === false && (
          <span className="hidden lg:inline text-ink-3" title="No hyperparameter search was run for any model in this bundle">
            · untuned
          </span>
        )}
      </div>
    </footer>
  );
}
