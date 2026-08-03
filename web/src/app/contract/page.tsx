"use client";

/**
 * What the dashboard reads out of a results bundle, and what the selected
 * bundle actually provides.
 *
 * This view exists because the two pipelines write different files. Rather
 * than let a reader guess why a column is empty on one dataset and full on
 * the other, the contract is stated and checked against the live bundle.
 */

import { useEffect, useState } from "react";
import { clsx } from "clsx";
import { AppShell } from "@/components/shell/AppShell";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { useBundle } from "@/components/bundle/BundleProvider";
import { type BundleDetail, type BundleMetrics, getBundle, isAbsent } from "@/lib/bundles";

/** Fields the UI asks for, and what it does when they are missing. */
const CONTRACT: {
  key: keyof BundleMetrics;
  need: "required" | "expected" | "optional";
  usedBy: string;
}[] = [
  { key: "accuracy", need: "required", usedBy: "Overview, Comparison" },
  { key: "f1_macro", need: "required", usedBy: "Comparison sort order" },
  { key: "f1_weighted", need: "expected", usedBy: "Comparison" },
  { key: "recall_macro", need: "expected", usedBy: "Comparison" },
  { key: "precision_macro", need: "optional", usedBy: "Model detail" },
  { key: "mcc", need: "expected", usedBy: "Comparison" },
  { key: "binary_fpr", need: "expected", usedBy: "Deployment view" },
  { key: "false_alarms_fp", need: "optional", usedBy: "Deployment view" },
  { key: "missed_attacks_fn", need: "optional", usedBy: "Deployment view" },
  { key: "throughput_flows_per_sec", need: "optional", usedBy: "Deployment view" },
  { key: "model_size_mb", need: "optional", usedBy: "Deployment view" },
  { key: "train_seconds", need: "optional", usedBy: "Cost table" },
  { key: "cv_f1_macro_mean", need: "optional", usedBy: "Stability panel" },
  { key: "label_shuffle_f1_macro", need: "optional", usedBy: "Leakage check" },
  { key: "f1_macro_ci_low", need: "optional", usedBy: "Significance panel" },
  { key: "hp_tuned", need: "optional", usedBy: "Fairness caveat" },
];

export default function ContractPage() {
  const { activeId } = useBundle();
  const [data, setData] = useState<BundleDetail | null>(null);

  useEffect(() => {
    if (!activeId) return;
    let cancelled = false;
    setData(null);
    getBundle(activeId)
      .then((d) => !cancelled && setData(d))
      .catch(() => !cancelled && setData(null));
    return () => {
      cancelled = true;
    };
  }, [activeId]);

  return (
    <AppShell title="Bundle contract">
      <div className="space-y-4">
        <Panel>
          <PanelHeader
            eyebrow={data?.dataset ?? "no bundle"}
            title="Fields the dashboard reads"
            sub={
              data
                ? `Checked against ${data.id} — a field counts as present when at least one model records it.`
                : "Select a bundle in the rail to check it against the contract."
            }
          />
          <div className="overflow-x-auto">
            <table className="w-full text-[11.5px]">
              <thead>
                <tr className="text-ink-3 border-b border-line-base">
                  <th className="text-left font-medium px-3 py-2">Field</th>
                  <th className="text-left font-medium px-3 py-2">Requirement</th>
                  <th className="text-left font-medium px-3 py-2">Used by</th>
                  <th className="text-left font-medium px-3 py-2">In this bundle</th>
                </tr>
              </thead>
              <tbody>
                {CONTRACT.map((row) => {
                  const present =
                    data != null &&
                    Object.values(data.models).some((m) => !isAbsent(m[row.key]));
                  return (
                    <tr key={String(row.key)} className="border-b border-line-subtle">
                      <td className="px-3 py-2 font-mono text-ink-0">{String(row.key)}</td>
                      <td className="px-3 py-2">
                        <Chip need={row.need} />
                      </td>
                      <td className="px-3 py-2 text-ink-2">{row.usedBy}</td>
                      <td className="px-3 py-2">
                        {data == null ? (
                          <span className="text-ink-3 font-mono">—</span>
                        ) : present ? (
                          <span className="text-ok font-mono text-[11px]">present</span>
                        ) : (
                          <span className="text-warn font-mono text-[11px]">absent</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel className="p-4">
          <h3 className="text-[12.5px] font-semibold text-ink-0">
            Why absent is not zero
          </h3>
          <p className="mt-1.5 text-[11.5px] text-ink-2 leading-relaxed max-w-prose">
            A false-positive rate of <span className="font-mono text-ink-1">0.00000</span> is
            the best score a detector can post. A false-positive rate that was never measured
            tells you nothing. Rendering the second as the first would make the least-evaluated
            model look like the best one, so every unrecorded field in this dashboard renders
            as <span className="font-mono text-ink-1">—</span> instead.
          </p>
        </Panel>
      </div>
    </AppShell>
  );
}

function Chip({ need }: { need: "required" | "expected" | "optional" }) {
  return (
    <span
      className={clsx(
        "px-1.5 py-0.5 rounded-sm text-[9.5px] font-mono uppercase tracking-wide border",
        need === "required" && "text-danger border-danger/30 bg-danger/10",
        need === "expected" && "text-warn border-warn/30 bg-warn/10",
        need === "optional" && "text-ink-3 border-line-base bg-surface",
      )}
    >
      {need}
    </span>
  );
}
