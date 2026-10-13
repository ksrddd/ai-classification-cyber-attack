"use client";

/**
 * Explainability — SHAP attributions, where a bundle has them.
 *
 * CICIDS2017 runs get SHAP from `main.py --stage explain`, CSE-CIC-IDS2018
 * runs from `scripts/shap_ids2018.py`. A model or bundle without output is
 * stated plainly, with the command that would produce it, rather than drawn
 * as an empty chart. Cross-dataset and audit bundles have no SHAP stage.
 */

import { useEffect, useState } from "react";
import { AppShell } from "@/components/shell/AppShell";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { BarRow } from "@/components/ui/BarRow";
import { BundleGate, Code, Empty } from "@/components/bundle/BundleGate";
import { modelColor } from "@/lib/colors";
import { type BundleDetail } from "@/lib/bundles";
import { getShap, type ShapPayload } from "@/lib/api";

/** Below this many rows a class's attribution is an anecdote, not a pattern. */
const MIN_CLASS_ROWS = 20;

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
  const [shap, setShap] = useState<ShapPayload | null>(null);
  const [missing, setMissing] = useState(false);
  const [cls, setCls] = useState("");
  const hasShapStage = data.layout === "cicids2017" || data.layout === "ids2018";

  useEffect(() => {
    if (!names.includes(selected)) setSelected(names[0] ?? "");
  }, [names, selected]);

  useEffect(() => {
    if (!selected || !hasShapStage) return;
    let cancelled = false;
    setShap(null);
    setMissing(false);
    // A model without SHAP output in this bundle surfaces as a 404.
    getShap(selected, data.id)
      .then((s) => {
        if (cancelled) return;
        setShap(s);
        const classes = Object.keys(s.per_class ?? {});
        setCls((c) => (classes.includes(c) ? c : classes[0] ?? ""));
      })
      .catch(() => !cancelled && setMissing(true));
    return () => {
      cancelled = true;
    };
  }, [selected, data.id, hasShapStage]);

  if (!hasShapStage) {
    return (
      <Empty title={`No SHAP analysis exists for ${data.id}`}>
        This bundle is not a single trained run, so there is no model to explain. This is
        stated rather than drawn as an empty chart, because an attribution plot with no
        data behind it invites the reader to interpret noise. Switch to a CICIDS2017 or
        CSE-CIC-IDS2018 run in the rail to see attributions.
      </Empty>
    );
  }

  const produce =
    data.layout === "ids2018"
      ? `python scripts/shap_ids2018.py --bundle ${data.id.split("/").pop()} --model ${selected}`
      : `python main.py --stage explain --model ${selected}`;
  const classRows = shap?.per_class?.[cls] ?? [];
  const classShare = shap?.per_class_share?.[cls];
  const classN = shap?.per_class_n?.[cls];
  const classPool = shap?.per_class_pool?.[cls];
  const classConf = shap?.per_class_confidence?.[cls];
  const nSeeds = shap?.seeds?.length ?? 1;

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
          sub={
            shap?.overall_share
              ? `Share of the model's total mean |SHAP|, class-balanced sample, mean ± SD over ${nSeeds} seeds. Shares compare across models; raw ${shap.units ?? ""} values do not.`
              : "Mean absolute SHAP value across the explained sample."
          }
        />
        {missing ? (
          <p className="px-3 pb-3 text-[11.5px] text-ink-2 max-w-prose leading-relaxed">
            No SHAP output is stored for {selected} in this run. Produce it with{" "}
            <Code>{produce}</Code>.
          </p>
        ) : shap === null ? (
          <div className="p-4 space-y-1.5">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="h-5 rounded-sm bg-surface animate-pulse" />
            ))}
          </div>
        ) : shap.overall_share ? (
          <>
            <Stability s={shap.stability?.overall} seeds={nSeeds} explainer={shap.explainer} />
            <ShareBars rows={shap.overall_share} color={modelColor(selected)} seeds={nSeeds} />
          </>
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

      {shap && classRows.length > 0 && (
        <Panel>
          <PanelHeader
            eyebrow={selected}
            title="Per-class feature importance"
            sub={
              classShare
                ? "Share of mean |SHAP| toward this class's score, over this class's own rows only."
                : "Mean absolute SHAP value toward one class's score."
            }
          />
          <div className="px-3 pb-2">
            <select
              value={cls}
              onChange={(e) => setCls(e.target.value)}
              className="bg-surface text-ink-1 border border-line-base rounded-sm px-2 py-1 text-[11.5px] font-mono"
            >
              {Object.keys(shap.per_class).map((c) => (
                <option key={c} value={c}>
                  {c}
                  {shap.per_class_n ? ` (n=${shap.per_class_n[c]})` : ""}
                </option>
              ))}
            </select>
          </div>
          {classPool && classPool.supplementary > 0 && (
            <p className="px-3 pb-2 text-[11.5px] text-ink-2 max-w-prose leading-relaxed">
              The locked test split holds only {classPool.test} {cls}{" "}
              {classPool.test === 1 ? "row" : "rows"}, so {classPool.supplementary} supplementary
              rows were added: flows of this class from the raw corpus that the training sample
              never drew, with exact duplicates of sampled rows removed. They are used for this
              explanation only and are not part of any reported metric.
            </p>
          )}
          {classN !== undefined && classN < MIN_CLASS_ROWS && (
            <p className="px-3 pb-2 text-[11.5px] text-warn max-w-prose leading-relaxed">
              Only {classN} {classN === 1 ? "row" : "rows"} of {cls} could be explained. Read this
              ranking as a description of those flows, not as what the model has learned about
              the class.
            </p>
          )}
          {classConf && classConf.mean_proba < 0.5 && (
            <p className="px-3 pb-2 text-[11.5px] text-warn max-w-prose leading-relaxed">
              On its own rows the model gives {cls} only{" "}
              {(classConf.mean_proba * 100).toFixed(0)}% probability on average (recall{" "}
              {(classConf.recall * 100).toFixed(0)}%). Attributions toward a class the model
              mostly misses are small, so an unstable ranking here means there is little to
              explain, not that the explainer failed.
            </p>
          )}
          {classShare ? (
            <>
              <Stability
                s={shap.stability?.per_class?.[cls]}
                seeds={nSeeds}
                explainer={shap.explainer}
              />
              <ShareBars rows={classShare.slice(0, 10)} color={modelColor(selected)} seeds={nSeeds} />
            </>
          ) : (
          <div className="px-3 pb-3 space-y-1">
            {classRows.slice(0, 10).map(([feature, value]) => (
              <BarRow
                key={feature}
                label={feature}
                value={value}
                max={classRows[0]?.[1] ?? 1}
                color={modelColor(selected)}
                suffix=""
                decimals={4}
              />
            ))}
          </div>
          )}
        </Panel>
      )}
    </div>
  );
}

function ShareBars({
  rows,
  color,
  seeds,
}: {
  rows: [string, number, number, boolean][];
  color: string;
  seeds: number;
}) {
  const max = rows[0]?.[1] ?? 1;
  return (
    <div className="px-3 pb-3 space-y-1">
      {rows.map(([feature, mean, sd, stable]) => (
        <BarRow
          key={feature}
          label={stable || seeds < 2 ? feature : `${feature}  ·  unstable`}
          value={mean * 100}
          max={max * 100}
          color={stable || seeds < 2 ? color : "rgb(var(--color-ink-3))"}
          suffix="%"
          decimals={1}
          right={
            seeds > 1 ? (
              <span className="tabular-nums text-[11px] text-ink-2 w-12 text-right">
                ±{(sd * 100).toFixed(1)}
              </span>
            ) : undefined
          }
        />
      ))}
    </div>
  );
}

function Stability({
  s,
  seeds,
  explainer,
}: {
  s?: Record<string, number | null>;
  seeds: number;
  explainer?: string;
}) {
  if (!s || seeds < 2) return null;
  // rho is null when a seed's attributions were all zero: rank order undefined.
  const rho = s.spearman_rho_min ?? null;
  const jac = s.top10_jaccard_min ?? 0;
  const ok = rho !== null && rho >= 0.9 && jac >= 0.8;
  return (
    <p className="px-3 pb-2 text-[11.5px] text-ink-2 max-w-prose leading-relaxed">
      <span className={ok ? "text-ok" : "text-warn"}>{ok ? "Stable" : "Unstable"}</span> across{" "}
      {seeds} seeds ({explainer} explainer): worst-pair Spearman ρ{" "}
      {rho === null ? "undefined (all-zero attributions)" : rho.toFixed(2)}, top-10
      overlap (Jaccard) {jac.toFixed(2)}. Greyed bars marked “unstable” left the top 10 on at
      least one seed.
    </p>
  );
}
