"use client";

/**
 * The cross-dataset view: what survives training on one corpus and testing on
 * the other.
 *
 * Built around a 2x2 rather than a ranked list, because the unit of result
 * here is a (train corpus, test corpus) pair. The diagonal is the ceiling —
 * what the model scores when it gets to train on the corpus it is judged on —
 * and the off-diagonal is the transfer being measured. Showing the two in one
 * grid is the point: a transfer number means nothing without the ceiling it is
 * a fraction of.
 *
 * The split-mode toggle is the other reason this page exists rather than a
 * static table in the report. Under a random split both directions read
 * 21-31% and look symmetric; under a chronological one they read 73.9% and
 * 7.5%. The finding is not that the numbers shift — it is that the wrong
 * split erases the asymmetry entirely, and a reader who can flip it themselves
 * does not have to take that on trust.
 */

import { useMemo, useState } from "react";
import { clsx } from "clsx";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { Nil } from "@/components/ui/Nil";
import { modelColor } from "@/lib/colors";
import {
  type CrossDataset,
  type TransferCell,
  count,
  isAbsent,
  score,
} from "@/lib/bundles";

const CORPUS_LABEL: Record<string, string> = {
  ids2017: "CICIDS2017",
  ids2018: "CSE-CIC-IDS2018",
};

const MODE_LABEL: Record<string, string> = {
  chronological: "Chronological",
  random: "Random",
};

function corpus(name: string) {
  return CORPUS_LABEL[name] ?? name;
}

export function TransferMatrix({ cross, id }: { cross: CrossDataset; id: string }) {
  // Chronological first when present: it is the split that does not let a
  // model see the back half of an attack window while being scored on the
  // front half, so it is the one the findings are stated on.
  const modes = useMemo(
    () => [...cross.modes].sort((a, b) => (a === "chronological" ? -1 : b === "chronological" ? 1 : 0)),
    [cross.modes],
  );
  const [mode, setMode] = useState(modes[0] ?? "chronological");
  const [model, setModel] = useState(() =>
    cross.models.includes("stacking") ? "stacking" : cross.models[0],
  );

  const corpora = useMemo(() => {
    const names: string[] = [];
    cross.cells.forEach((c) => {
      if (!names.includes(c.train)) names.push(c.train);
      if (!names.includes(c.test)) names.push(c.test);
    });
    return names.sort();
  }, [cross.cells]);

  const cellAt = (train: string, test: string) =>
    cross.cells.find(
      (c) => c.mode === mode && c.model === model && c.train === train && c.test === test,
    ) ?? null;

  const transfers = cross.cells.filter(
    (c) => c.mode === mode && !c.self && c.model === model,
  );

  return (
    <div className="space-y-4">
      <Panel>
        <PanelHeader
          eyebrow={id}
          title="Cross-dataset transfer"
          sub="Trained on one corpus, scored on the other. Nothing is adapted to the target."
          right={
            <div className="flex gap-1">
              {modes.map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  aria-pressed={m === mode}
                  className={clsx(
                    "px-2 py-1 rounded-sm text-[10px] font-mono uppercase tracking-wide border transition-colors",
                    m === mode
                      ? "bg-info/15 border-info/40 text-info"
                      : "bg-surface-elevated border-line-base text-ink-2 hover:text-ink-0",
                  )}
                >
                  {MODE_LABEL[m] ?? m} split
                </button>
              ))}
            </div>
          }
        />

        <div className="px-4 py-3 space-y-3">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[10px] uppercase tracking-[.18em] text-ink-3 font-semibold mr-1">
              Model
            </span>
            {cross.models.map((m) => (
              <button
                key={m}
                onClick={() => setModel(m)}
                aria-pressed={m === model}
                className={clsx(
                  "px-2 py-0.5 rounded-sm text-[10.5px] font-mono border transition-colors",
                  m === model
                    ? "border-line-base text-ink-0 bg-surface-elevated"
                    : "border-transparent text-ink-2 hover:text-ink-0",
                )}
                style={m === model ? { borderColor: modelColor(m) } : undefined}
              >
                {m}
              </button>
            ))}
          </div>

          {/* The matrix. Rows are what the model was trained on, columns what
              it was scored on, so reading across a row answers "where does
              this training set take you". */}
          <div className="overflow-x-auto">
            <table className="w-full min-w-[520px] border-separate border-spacing-1">
              <thead>
                <tr>
                  <th className="w-32" />
                  {corpora.map((c) => (
                    <th
                      key={c}
                      scope="col"
                      className="text-[10px] uppercase tracking-[.14em] text-ink-3 font-semibold pb-1"
                    >
                      tested on {corpus(c)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {corpora.map((train) => (
                  <tr key={train}>
                    <th
                      scope="row"
                      className="text-right pr-2 text-[10px] uppercase tracking-[.14em] text-ink-3 font-semibold align-middle"
                    >
                      trained on
                      <br />
                      <span className="text-ink-1">{corpus(train)}</span>
                    </th>
                    {corpora.map((test) => (
                      <td key={test} className="align-top">
                        <Cell cell={cellAt(train, test)} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="text-[11px] text-ink-2 leading-relaxed max-w-prose">
            Diagonal cells are the ceiling — the model trained and tested on the same
            corpus. Off-diagonal cells never saw the corpus they are scored on, not even
            to fit the scaler. <strong className="text-ink-1">Recovered</strong> is the
            share of the distance from the majority-class baseline to that ceiling which
            the transfer closed, which is what makes two models with different ceilings
            comparable.
          </p>
        </div>
      </Panel>

      <div className="grid lg:grid-cols-2 gap-4">
        <RecoveredPanel cross={cross} mode={mode} />
        <LeakagePanel cross={cross} />
      </div>

      <PerClassPanel cross={cross} mode={mode} model={model} transfers={transfers} />
      <CorporaPanel cross={cross} />
    </div>
  );
}

function Cell({ cell }: { cell: TransferCell | null }) {
  if (!cell) {
    return (
      <div className="rounded border border-line-subtle bg-surface-elevated/40 p-3 text-center">
        <Nil />
      </div>
    );
  }

  const recovered = cell.recovered_pct;
  // Bands, not a gradient: the question a reader has is which side of "worth
  // anything" a cell falls on. 25% and 60% are labels for the three answers
  // the run produced, not thresholds anything is selected by.
  const tone = cell.self
    ? "ceiling"
    : isAbsent(recovered)
      ? "none"
      : (recovered as number) >= 60
        ? "good"
        : (recovered as number) >= 25
          ? "partial"
          : "poor";

  return (
    <div
      className={clsx(
        "rounded border p-3 h-full",
        tone === "ceiling" && "border-line-base bg-surface-elevated",
        tone === "good" && "border-ok/40 bg-ok/10",
        tone === "partial" && "border-warn/40 bg-warn/10",
        tone === "poor" && "border-danger/40 bg-danger/10",
        tone === "none" && "border-line-subtle bg-surface-elevated/40",
      )}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span
          className={clsx(
            "text-[9.5px] uppercase tracking-[.16em] font-semibold",
            cell.self ? "text-ink-3" : "text-ink-2",
          )}
        >
          {cell.self ? "ceiling" : "transfer"}
        </span>
        {!isAbsent(cell.sd) && (
          <span className="text-[9.5px] font-mono text-ink-3" title="Spread across the protocol's five seeds">
            ±{score(cell.sd, 3)}
          </span>
        )}
      </div>

      <div className="mt-1 font-mono tabular-nums text-[19px] text-ink-0 leading-none">
        {score(cell.f1_macro, 4)}
      </div>
      <div className="text-[9.5px] text-ink-3 mt-0.5">macro-F1</div>

      {!cell.self && (
        <div className="mt-2 pt-2 border-t border-line-subtle space-y-1">
          <div className="flex justify-between text-[10.5px] font-mono">
            <span className="text-ink-3">recovered</span>
            <span
              className={clsx(
                "tabular-nums font-semibold",
                tone === "good" && "text-ok",
                tone === "partial" && "text-warn",
                tone === "poor" && "text-danger",
              )}
            >
              {isAbsent(recovered) ? <Nil /> : `${(recovered as number).toFixed(1)}%`}
            </span>
          </div>
          <div className="flex justify-between text-[10px] font-mono text-ink-3">
            <span>baseline {score(cell.floor, 3)}</span>
            <span>ceiling {score(cell.ceiling, 3)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

function RecoveredPanel({ cross, mode }: { cross: CrossDataset; mode: string }) {
  const rows = cross.cells
    .filter((c) => c.mode === mode && !c.self)
    .sort((a, b) => (b.recovered_pct ?? -1) - (a.recovered_pct ?? -1));

  return (
    <Panel>
      <PanelHeader
        title="Every model, both directions"
        sub={`${MODE_LABEL[mode] ?? mode} split · share of the baseline-to-ceiling range recovered`}
      />
      <div className="px-4 pb-3 pt-2 space-y-1">
        {rows.map((c) => (
          <div key={`${c.model}-${c.train}-${c.test}`} className="flex items-center gap-2 text-[11px]">
            <span
              className="h-2 w-2 rounded-full flex-shrink-0"
              style={{ background: modelColor(c.model) }}
            />
            <span className="w-36 truncate text-ink-1">{c.model}</span>
            <span className="w-24 font-mono text-[9.5px] text-ink-3 tabular-nums">
              {c.train.replace("ids", "")}→{c.test.replace("ids", "")}
            </span>
            <div className="flex-1 h-1.5 rounded-full bg-surface-elevated overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${Math.max(0, Math.min(100, c.recovered_pct ?? 0))}%`,
                  background: modelColor(c.model),
                }}
              />
            </div>
            <span className="w-12 text-right font-mono tabular-nums text-ink-1">
              {isAbsent(c.recovered_pct) ? <Nil /> : `${(c.recovered_pct as number).toFixed(1)}%`}
            </span>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function LeakagePanel({ cross }: { cross: CrossDataset }) {
  const rows = [...cross.leakage].sort((a, b) => (b.inflation ?? 0) - (a.inflation ?? 0));

  return (
    <Panel>
      <PanelHeader
        title="What the random split adds"
        sub="Same corpus, same model, ceiling under each split mode"
      />
      <div className="overflow-x-auto">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="text-[9.5px] uppercase tracking-[.14em] text-ink-3">
              <th className="text-left font-semibold px-4 py-1.5">corpus · model</th>
              <th className="text-right font-semibold px-2">temporal</th>
              <th className="text-right font-semibold px-2">random</th>
              <th className="text-right font-semibold px-4">inflation</th>
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {rows.map((r) => (
              <tr key={`${r.test}-${r.model}`} className="border-t border-line-subtle">
                <td className="px-4 py-1 text-ink-1">
                  <span className="text-ink-3">{corpus(r.test)}</span> · {r.model}
                </td>
                <td className="px-2 text-right text-ink-1">{score(r.chronological, 4)}</td>
                <td className="px-2 text-right text-ink-1">{score(r.random, 4)}</td>
                <td
                  className={clsx(
                    "px-4 text-right font-semibold",
                    (r.inflation ?? 0) >= 0.1 ? "text-danger" : "text-ink-2",
                  )}
                >
                  +{score(r.inflation, 4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="px-4 py-2.5 text-[11px] text-ink-2 leading-relaxed">
        The inflation is an order of magnitude larger on one corpus than the other, from
        the split alone — so a random split does not merely make both ceilings optimistic,
        it makes them optimistic by different amounts and breaks the comparison between them.
      </p>
    </Panel>
  );
}

function PerClassPanel({
  cross,
  mode,
  model,
  transfers,
}: {
  cross: CrossDataset;
  mode: string;
  model: string;
  transfers: TransferCell[];
}) {
  const rows = cross.per_class.filter((r) => r.mode === mode && r.model === model);
  const support = cross.support.filter((s) => s.mode === mode);
  const supportFor = (test: string) =>
    support.find((s) => s.test === test)?.per_class ?? {};

  return (
    <Panel>
      <PanelHeader
        title="Which classes survive the crossing"
        sub={`${model} · ${MODE_LABEL[mode] ?? mode} split · per-class F1, with test rows behind each score`}
      />
      <div className="overflow-x-auto">
        <table className="w-full text-[11px] min-w-[640px]">
          <thead>
            <tr className="text-[9.5px] uppercase tracking-[.14em] text-ink-3">
              <th className="text-left font-semibold px-4 py-1.5">train → test</th>
              {cross.classes.map((c) => (
                <th key={c} className="text-right font-semibold px-2">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="font-mono tabular-nums">
            {rows.map((r) => {
              const n = supportFor(r.test);
              return (
                <tr
                  key={`${r.train}-${r.test}`}
                  className={clsx(
                    "border-t border-line-subtle",
                    r.self && "bg-surface-elevated/40",
                  )}
                >
                  <td className="px-4 py-1 whitespace-nowrap">
                    <span className="text-ink-1">
                      {corpus(r.train)} → {corpus(r.test)}
                    </span>
                    {r.self && (
                      <span className="ml-2 text-[9px] uppercase tracking-wide text-ink-3">
                        ceiling
                      </span>
                    )}
                  </td>
                  {cross.classes.map((c) => {
                    const v = r.f1[c];
                    const rows_n = n[c];
                    // A class with a handful of test rows cannot support an F1
                    // that describes the model, so the count travels with the
                    // score rather than sitting in a separate table nobody
                    // cross-references.
                    const thin = !isAbsent(rows_n) && (rows_n as number) < 50;
                    return (
                      <td key={c} className="px-2 text-right align-top py-1">
                        <div
                          className={clsx(
                            isAbsent(v)
                              ? "text-ink-3"
                              : (v as number) >= 0.6
                                ? "text-ok"
                                : (v as number) >= 0.2
                                  ? "text-warn"
                                  : "text-danger",
                          )}
                        >
                          {score(v, 3)}
                        </div>
                        <div
                          className={clsx(
                            "text-[9px]",
                            thin ? "text-danger/80" : "text-ink-3",
                          )}
                          title={
                            thin
                              ? "Too few test rows for this F1 to describe the model"
                              : undefined
                          }
                        >
                          n={count(rows_n)}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {transfers.length > 0 && (
        <p className="px-4 py-2.5 text-[11px] text-ink-2 leading-relaxed max-w-prose">
          A macro average over seven classes hides that one of them may be carrying all of
          it. Classes marked with a small <span className="text-danger/80 font-mono">n</span>{" "}
          have too few test rows for their F1 to describe the model rather than the handful
          of flows that happened to land in the test split.
        </p>
      )}
    </Panel>
  );
}

function CorporaPanel({ cross }: { cross: CrossDataset }) {
  const entries = Object.entries(cross.corpora);
  if (!entries.length) return null;

  return (
    <Panel>
      <PanelHeader
        title="What entered the run"
        sub="Both corpora pass the same cleaning rules and the same 77-feature schema"
      />
      <div className="grid sm:grid-cols-2 gap-px bg-line-subtle">
        {entries.map(([name, c]) => {
          const cleaning = (c.cleaning ?? {}) as Record<string, unknown>;
          const classes = (c.class_counts ?? {}) as Record<string, number>;
          return (
            <div key={name} className="bg-surface-raised p-4 space-y-2">
              <div className="text-[12px] font-semibold text-ink-0">{corpus(name)}</div>
              <dl className="text-[11px] space-y-1 font-mono">
                <Row k="rows after cleaning" v={count(c.n_rows as number)} />
                <Row k="features" v={count(c.n_features as number)} />
                <Row k="ordered by" v={String(c.order_basis ?? "—")} />
                <Row k="dropped, unmapped label" v={count(c.dropped_unmapped_rows as number)} />
                <Row k="dropped, duplicate" v={count(cleaning.rows_dropped_duplicate as number)} />
                <Row k="dropped, non-finite" v={count(cleaning.rows_dropped_non_finite as number)} />
              </dl>
              <div className="pt-1 space-y-0.5">
                {Object.entries(classes).map(([cls, n]) => (
                  <div key={cls} className="flex justify-between text-[10.5px] font-mono">
                    <span className="text-ink-2">{cls}</span>
                    <span className="text-ink-1 tabular-nums">{count(n)}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-ink-3">{k}</dt>
      <dd className="text-ink-1 tabular-nums">{v}</dd>
    </div>
  );
}
