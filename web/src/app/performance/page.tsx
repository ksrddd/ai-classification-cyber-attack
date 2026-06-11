"use client";

import { useState, useEffect } from "react";
import { AppShell } from "@/components/shell/AppShell";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { KpiCard } from "@/components/ui/KpiCard";
import { BarRow } from "@/components/ui/BarRow";
import { Pill } from "@/components/ui/Pill";
import { getModels, getModelMetrics, getModelReport, figureUrl } from "@/lib/api";
import { modelColor, modelLabel } from "@/lib/colors";

const METRIC_DISPLAY = [
  { k: "accuracy",            l: "Accuracy",    color: "#38BDF8" },
  { k: "precision_weighted",  l: "Precision",   color: "#3B82F6" },
  { k: "recall_weighted",     l: "Recall",      color: "#F43F5E" },
  { k: "f1_weighted",         l: "F1 (W)",      color: "#22D3EE" },
  { k: "f1_macro",            l: "F1 (M)",      color: "#6366F1" },
  { k: "roc_auc",             l: "ROC-AUC",     color: "#10B981" },
];

function modelShort(name: string): string {
  const map: Record<string, string> = {
    random_forest: "RF", xgboost: "XGB", lightgbm: "LGB",
    catboost: "CB", mlp: "MLP", logistic_regression: "LR",
  };
  return map[name] ?? name.slice(0, 3).toUpperCase();
}

export default function PerformancePage() {
  const [models, setModels] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [metrics, setMetrics] = useState<Record<string, number> | null>(null);
  const [report, setReport] = useState<Record<string, Record<string, number>> | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getModels().then((d) => {
      setModels(d.models);
      if (d.models[0]) setSelected(d.models[0]);
    });
  }, []);

  useEffect(() => {
    if (!selected) return;
    setLoading(true);
    Promise.all([
      getModelMetrics(selected).catch(() => null),
      getModelReport(selected).catch(() => null),
    ]).then(([m, r]) => {
      setMetrics(m as Record<string, number> | null);
      setReport(r);
      setLoading(false);
    });
  }, [selected]);

  const color = modelColor(selected);
  const m = metrics ?? {};

  const perClass = (m as Record<string, unknown>).per_class as
    Record<string, { precision: number; recall: number; f1: number }> | undefined;

  return (
    <AppShell title="Model Performance">
      <div className="space-y-5 animate-fadeIn">

        {/* Header */}
        <div className="flex items-end justify-between flex-wrap gap-3">
          <div>
            <h1 className="text-[26px] font-semibold tracking-tight text-ink-0">Model Performance</h1>
            <p className="text-[12px] text-ink-2 mt-1">
              Per-model metrics · confusion matrix · per-class breakdown
            </p>
          </div>
          {/* Model selector */}
          <div className="flex flex-wrap gap-1.5">
            {models.map((name) => {
              const c = modelColor(name);
              const isActive = name === selected;
              return (
                <button
                  key={name}
                  onClick={() => setSelected(name)}
                  className="px-3 h-8 rounded-lg text-[12px] font-medium transition ring-1"
                  style={{
                    background: isActive ? `${c}22` : "transparent",
                    color: isActive ? c : "#6C7488",
                    borderColor: isActive ? `${c}55` : "rgba(255,255,255,0.08)",
                    boxShadow: isActive ? `0 0 12px ${c}33` : "none",
                  }}
                >
                  {modelLabel(name)}
                </button>
              );
            })}
          </div>
        </div>

        {loading ? (
          <div className="text-center py-16 text-ink-2 text-[12px]">
            <span className="inline-flex items-center gap-2">
              <span className="h-1.5 w-1.5 rounded-full bg-info animate-pulseDot" />
              Loading metrics…
            </span>
          </div>
        ) : (
          <>
            {/* Champion banner */}
            {selected && (
              <div
                className="relative overflow-hidden rounded-xl p-4 ring-1"
                style={{
                  background: `${color}0A`,
                  borderColor: `${color}30`,
                }}
              >
                <div
                  className="absolute -right-8 -top-8 h-32 w-32 rounded-full blur-2xl opacity-20 pointer-events-none"
                  style={{ background: color }}
                />
                <div className="relative flex items-center gap-4 flex-wrap">
                  <div
                    className="h-12 w-12 rounded-xl grid place-items-center font-semibold text-[14px] flex-shrink-0"
                    style={{ background: `${color}1F`, color, boxShadow: `inset 0 0 0 1px ${color}55` }}
                  >
                    {modelShort(selected)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-[16px] font-semibold text-ink-0">{modelLabel(selected)}</span>
                      <Pill tone="brand" size="sm">holdout test set</Pill>
                    </div>
                    <div className="text-[11px] text-ink-2 mt-0.5 tabular-nums">
                      F1 = {(m.f1_weighted ?? 0).toFixed(4)} · Accuracy = {(m.accuracy ?? 0).toFixed(4)} · MCC = {(m.matthews_corrcoef ?? 0).toFixed(4)}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* KPI tiles */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <KpiCard label="Accuracy"      value={(m.accuracy      ?? 0).toFixed(4)} color={color} />
              <KpiCard label="F1 (weighted)" value={(m.f1_weighted   ?? 0).toFixed(4)} color={color} />
              <KpiCard label="F1 (macro)"    value={(m.f1_macro      ?? 0).toFixed(4)} color={color} />
              <KpiCard label="ROC-AUC"       value={(m.roc_auc       ?? 0).toFixed(4)} color={color} />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <KpiCard label="Precision (W)" value={(m.precision_weighted ?? 0).toFixed(4)} color={color} />
              <KpiCard label="Recall (W)"    value={(m.recall_weighted    ?? 0).toFixed(4)} color={color} />
              <KpiCard label="Precision (M)" value={(m.precision_macro    ?? 0).toFixed(4)} color={color} />
              <KpiCard label="MCC"           value={(m.matthews_corrcoef  ?? 0).toFixed(4)} color={color} />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Metric fingerprint */}
              <Panel>
                <PanelHeader eyebrow="Radar" title="Metric fingerprint" sub="Weighted + macro scores" />
                <div className="px-5 py-4 space-y-3">
                  {METRIC_DISPLAY.map(({ k, l, color: c }) => (
                    <BarRow key={k} label={l} value={(m[k] ?? 0) * 100} max={100} color={c} suffix="%" />
                  ))}
                </div>
              </Panel>

              {/* Confusion matrix */}
              <Panel>
                <PanelHeader
                  eyebrow="Evaluation"
                  title="Confusion matrix"
                  sub="Normalised by true class"
                />
                <div className="p-4">
                  <img
                    src={figureUrl(`confusion_matrix_${selected}.png`)}
                    alt="Confusion matrix"
                    className="w-full rounded-lg"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = "none";
                    }}
                  />
                </div>
              </Panel>
            </div>

            {/* Per-class breakdown */}
            {perClass && Object.keys(perClass).length > 0 && (
              <Panel>
                <PanelHeader
                  eyebrow="Per-class"
                  title="Precision / Recall / F1 by attack family"
                  right={<Pill tone="muted" size="sm">holdout fold</Pill>}
                />
                <div className="overflow-x-auto">
                  <table className="w-full text-[12px]">
                    <thead>
                      <tr className="text-left border-b border-line-subtle">
                        {["Class", "Precision", "Recall", "F1-score", "Support"].map((h) => (
                          <th
                            key={h}
                            className="px-4 py-2.5 text-[10px] uppercase tracking-[.14em] text-ink-2 font-medium"
                          >
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(perClass).map(([cls, vals]) => {
                        const f1 = vals.f1 ?? 0;
                        const textColor =
                          f1 >= 0.95 ? "#10B981" : f1 >= 0.85 ? "#E6E9F2" : "#F59E0B";
                        return (
                          <tr
                            key={cls}
                            className="border-b border-line-subtle last:border-0 hover:bg-white/[.02]"
                          >
                            <td className="px-4 py-2.5 text-ink-0 font-medium">{cls}</td>
                            {(["precision", "recall", "f1"] as const).map((k) => (
                              <td key={k} className="px-4 py-2.5">
                                <div className="flex items-center gap-2">
                                  <span
                                    className="tabular-nums"
                                    style={{ color: textColor }}
                                  >
                                    {(vals[k] ?? 0).toFixed(4)}
                                  </span>
                                  <div className="hidden sm:block flex-1 h-0.5 rounded-full bg-white/[.04] overflow-hidden max-w-[60px]">
                                    <div
                                      className="h-full rounded-full"
                                      style={{
                                        width: `${(vals[k] ?? 0) * 100}%`,
                                        background: textColor,
                                      }}
                                    />
                                  </div>
                                </div>
                              </td>
                            ))}
                            <td className="px-4 py-2.5 tabular-nums text-ink-2">
                              {(vals as Record<string, number>).support
                                ? Math.round((vals as Record<string, number>).support).toLocaleString()
                                : "—"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </Panel>
            )}
          </>
        )}
      </div>
    </AppShell>
  );
}
