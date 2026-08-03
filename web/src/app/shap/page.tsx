"use client";

/**
 * Explainability — SHAP attributions, where a bundle has them.
 *
 * Only the CICIDS2017 pipeline runs a SHAP stage. Rather than hide this page
 * or show an empty chart on CSE-CIC-IDS2018, it states plainly that the
 * analysis was never computed for the selected run and what would produce it.
 */

import { useEffect, useState } from "react";
import { AppShell } from "@/components/shell/AppShell";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { BarRow } from "@/components/ui/BarRow";
import { BundleGate, Code, Empty } from "@/components/bundle/BundleGate";
import { modelColor } from "@/lib/colors";
import { type BundleDetail } from "@/lib/bundles";
import { getShap } from "@/lib/api";

export default function ShapPage() {
  return (
    <AppShell title="Explainability">
      <BundleGate what="explainability">{(data) => <Body data={data} />}</BundleGate>
    </AppShell>
  );
}

function Body({ data }: { data: BundleDetail }) {
  const names = Object.keys(data.models).sort();
  const [selected, setSelected] = useState(names[0] ?? "");
  const [shap, setShap] = useState<{ overall: [string, number][] } | null>(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    if (!names.includes(selected)) setSelected(names[0] ?? "");
  }, [names, selected]);

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    setShap(null);
    setMissing(false);
    // /api/shap reads the published 2017 run; any other bundle has no SHAP
    // stage at all, which surfaces here as a 404.
    getShap(selected)
      .then((s) => !cancelled && setShap(s))
      .catch(() => !cancelled && setMissing(true));
    return () => {
      cancelled = true;
    };
  }, [selected]);

  if (data.layout !== "cicids2017") {
    return (
      <Empty title={`No SHAP analysis exists for ${data.id}`}>
        The CSE-CIC-IDS2018 pipeline has no explainability stage, so nothing was ever
        computed for this bundle. This is stated rather than drawn as an empty chart,
        because an attribution plot with no data behind it invites the reader to interpret
        noise. Switch to a CICIDS2017 run in the rail to see attributions, or add a SHAP
        step to <Code>src/ids2018/</Code> to produce them here.
      </Empty>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-1.5">
        {names.map((n) => (
          <button
            key={n}
            onClick={() => setSelected(n)}
            className={
              "px-2.5 py-1 rounded-sm text-[11px] font-mono border transition-colors " +
              (n === selected
                ? "bg-surface-elevated text-ink-0 border-info"
                : "bg-surface text-ink-2 border-line-base hover:text-ink-1")
            }
          >
            {n}
          </button>
        ))}
      </div>

      <Panel>
        <PanelHeader
          eyebrow={selected}
          title="Global feature importance"
          sub="Mean absolute SHAP value across the explained sample."
        />
        {missing ? (
          <p className="px-3 pb-3 text-[11.5px] text-ink-2 max-w-prose leading-relaxed">
            No SHAP output is stored for {selected} in this run. Produce it with{" "}
            <Code>python main.py --stage explain --model {selected}</Code>.
          </p>
        ) : shap === null ? (
          <div className="p-4 space-y-1.5">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="h-5 rounded-sm bg-surface animate-pulse" />
            ))}
          </div>
        ) : (
          <div className="px-3 pb-3 space-y-1">
            {shap.overall.slice(0, 15).map(([feature, value]) => (
              <BarRow
                key={feature}
                label={feature}
                value={value}
                max={shap.overall[0]?.[1] ?? 1}
                color={modelColor(selected)}
                suffix=""
                decimals={4}
              />
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}
