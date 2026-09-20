"use client";

/**
 * The rail's run picker, plus the provenance the choice implies.
 *
 * The mockup puts this at the top of the sidebar for a reason: which bundle
 * is on screen is a decision, not a hidden default. The two bundles disagree
 * about the split protocol, the class count and whether the models were
 * tuned, so every number below changes meaning when this changes.
 */

import { clsx } from "clsx";
import { AlertTriangle, Database } from "lucide-react";
import { useBundle } from "./BundleProvider";
import { NIL, count } from "@/lib/bundles";

/** What each audit asks, for the one line the rail has room for. */
const AUDIT_ASKS: Record<string, string> = {
  adversarial_validation: "which columns name the corpus",
  feature_mapping: "do the 77 columns still pair",
  stacking_leakage: "can stacking see its test set",
  ablation: "does dropping them help",
};

export function RunSelector({ collapsed }: { collapsed: boolean }) {
  const { bundles, active, select, loading, error } = useBundle();

  if (collapsed) {
    return (
      <div className="px-3.5 py-2.5 border-b border-line-subtle grid place-items-center">
        <Database size={14} className={active ? "text-info" : "text-ink-3"} />
      </div>
    );
  }

  if (loading) {
    return (
      <div className="px-4 py-3 border-b border-line-subtle">
        <div className="h-7 rounded-sm bg-surface-raised animate-pulse" />
      </div>
    );
  }

  if (error || !active) {
    return (
      <div className="px-4 py-3 border-b border-line-subtle">
        <div className="flex items-start gap-2 text-[11px] text-warn">
          <AlertTriangle size={13} className="flex-shrink-0 mt-px" />
          <span className="leading-snug">
            {error ? "API unreachable" : "No results bundle found"}
          </span>
        </div>
      </div>
    );
  }

  // Split protocol decides how the numbers may be read, so it is surfaced
  // here rather than buried in a provenance page.
  const temporal = active.split_protocol?.includes("temporal");
  // A cross-dataset run carries both split modes and is switched between them
  // in the view itself, so the single split badge below would have to pick one
  // and label the run wrongly either way. It also has no one train/test row
  // count: each corpus has its own. Both get their own treatment rather than
  // being forced through fields that assume a single training split.
  const cross = active.layout === "crossdataset";
  // An audit has no split, no tuning and no models. Every badge below would
  // have to invent an answer for it, and "untuned" on a run that trained
  // nothing reads as a property of a model that does not exist.
  const audit = active.layout === "audit";

  return (
    <div className="px-4 py-3 border-b border-line-subtle space-y-2">
      <label
        htmlFor="run-select"
        className="block text-[9px] uppercase tracking-[.2em] text-ink-3 font-semibold"
      >
        Active run
      </label>

      <select
        id="run-select"
        value={active.id}
        onChange={(e) => select(e.target.value)}
        className={clsx(
          "w-full h-7 px-2 rounded-sm text-[11.5px] font-mono",
          "bg-surface-raised border border-line-base text-ink-0",
          "focus:outline-none focus:border-info",
        )}
      >
        {bundles.map((b) => (
          <option key={b.id} value={b.id}>
            {b.id}
          </option>
        ))}
      </select>

      <div className="text-[10.5px] text-ink-2 leading-relaxed font-mono">
        <div className="text-ink-1">{active.dataset}</div>
        <div>
          {audit
            ? AUDIT_ASKS[active.audit_kind ?? ""] ?? "verification check"
            : cross
              ? `${active.n_models} models × 4 directions`
              : `${count(active.n_train)} train · ${count(active.n_test)} test`}
        </div>
        {(!audit || active.n_classes > 0) && (
          <div>
            {active.n_classes} {cross || audit ? "shared classes" : "classes"} ·{" "}
            {active.n_features ?? NIL} features
          </div>
        )}
      </div>

      <div className="flex flex-wrap gap-1">
        {audit ? (
          <Badge
            tone="ok"
            label="audit"
            title="Not a training run: this bundle checks whether an existing result can be believed. It scores no models, so there is no split protocol and no champion."
          />
        ) : cross ? (
          <Badge
            tone="ok"
            label="both splits"
            title="A cross-dataset run scores every combination under both a chronological and a random split; pick between them in the view."
          />
        ) : (
          <Badge
            tone={temporal ? "ok" : "warn"}
            label={temporal ? "temporal split" : "random split"}
            title={
              temporal
                ? "Chronological split — no flow from the same attack burst spans train and test."
                : "Random stratified split — near-duplicate flows can span train and test, so scores are optimistic."
            }
          />
        )}
        {!audit && (
        <Badge
          tone={active.hp_tuned ? "ok" : "muted"}
          label={active.hp_tuned ? "tuned" : "untuned"}
          title={
            active.hp_tuned
              ? "Hyperparameters were searched for at least some models."
              : cross
                ? "Library defaults on both sides, by design: tuning moves these models by 0.01–0.05 against a transfer gap of 0.4–0.6, and importing one corpus's tuned parameters would hand it an advantage shaped like a dataset effect."
                : "No hyperparameter search was run; every model uses fixed defaults."
          }
        />
        )}
      </div>
    </div>
  );
}

function Badge({
  tone,
  label,
  title,
}: {
  tone: "ok" | "warn" | "muted";
  label: string;
  title: string;
}) {
  return (
    <span
      title={title}
      className={clsx(
        "px-1.5 py-0.5 rounded-sm text-[9.5px] font-mono uppercase tracking-wide border cursor-help",
        tone === "ok" && "text-ok border-ok/30 bg-ok/10",
        tone === "warn" && "text-warn border-warn/30 bg-warn/10",
        tone === "muted" && "text-ink-3 border-line-base bg-surface-raised",
      )}
    >
      {label}
    </span>
  );
}
