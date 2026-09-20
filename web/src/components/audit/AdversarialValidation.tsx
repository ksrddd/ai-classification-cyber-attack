"use client";

/**
 * Adversarial validation: can a classifier name the corpus a row came from?
 *
 * The number this view exists to make readable is an AUC of 1.0, which is
 * exactly the number nobody should accept on sight. So the page is ordered as
 * the argument was made rather than by importance: the result first, then
 * every attempt that was made to break it, and only then the feature ranking
 * — which is the part that would be worthless if the attempts had succeeded.
 *
 * High is bad here, and the colour scale says so. Everywhere else in this
 * dashboard a high score is a working detector; here it means the two corpora
 * are trivially distinguishable, which is the finding that makes cross-dataset
 * transfer fail. Chance (0.50) is the good outcome.
 */

import { useMemo, useState } from "react";
import { clsx } from "clsx";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { Nil } from "@/components/ui/Nil";
import { Pill } from "@/components/ui/Pill";
import {
  type AdversarialAudit,
  type AdversarialDiagnosticClass,
  type WithinDatasetControl,
  count,
  isAbsent,
  percent,
  score,
} from "@/lib/bundles";

const CORPUS_LABEL: Record<string, string> = {
  ids2017: "CICIDS2017",
  ids2018: "CSE-CIC-IDS2018",
};

/** The rungs of the capacity ladder, weakest first. */
const LADDER: { key: string; label: string; hint: string }[] = [
  {
    key: "stump",
    label: "stump",
    hint: "Decision tree of depth 1 — one threshold on one column. It cannot memorise rows.",
  },
  { key: "tree_depth3", label: "tree (3)", hint: "Decision tree of depth 3 — at most eight leaves." },
  { key: "logreg", label: "logreg", hint: "Logistic regression on standardised columns." },
  {
    key: "logreg_rank",
    label: "logreg (rank)",
    hint: "Logistic regression after a rank transform, so only the ordering of each column survives.",
  },
  { key: "lightgbm", label: "lightgbm", hint: "The 300-tree ensemble used for the headline AUC." },
];

/**
 * Bands, not a gradient. The reader's question is which of four things a cell
 * says: chance, a hint, most of the way, or fully separable.
 */
function aucTone(v: number | null | undefined) {
  if (isAbsent(v)) return "none" as const;
  const x = v as number;
  if (x >= 0.99) return "sep" as const;
  if (x >= 0.9) return "high" as const;
  if (x >= 0.7) return "mid" as const;
  return "chance" as const;
}

const TONE_CLASS: Record<string, string> = {
  sep: "border-danger/40 bg-danger/15 text-danger",
  high: "border-warn/40 bg-warn/15 text-warn",
  mid: "border-info/40 bg-info/10 text-info",
  chance: "border-ok/40 bg-ok/10 text-ok",
  none: "border-line-subtle bg-surface-elevated/40 text-ink-3",
};

export function AdversarialValidation({
  audit,
  id,
}: {
  audit: AdversarialAudit;
  id: string;
}) {
  const diagByClass = useMemo(() => {
    const out = new Map<string, AdversarialDiagnosticClass>();
    audit.diagnostics?.classes.forEach((c) => out.set(c.shared_class, c));
    return out;
  }, [audit.diagnostics]);

  const aucs = audit.classes.map((c) => c.auc).filter((v): v is number => !isAbsent(v));
  const lo = aucs.length ? Math.min(...aucs) : null;
  const hi = aucs.length ? Math.max(...aucs) : null;

  return (
    <div className="space-y-4">
      <Panel>
        <PanelHeader
          eyebrow={id}
          title="What tells the two corpora apart?"
          sub="One class at a time, balanced, 2017 = 0 and 2018 = 1. Whatever the classifier finds useful describes the lab rather than the attack."
          right={
            <Pill tone={hi !== null && hi >= 0.99 ? "danger" : "muted"} size="sm">
              {lo === null ? "no result" : `AUC ${score(lo, 4)}–${score(hi, 4)}`}
            </Pill>
          }
        />
        <div className="px-4 py-3 space-y-3">
          <div className="flex flex-wrap gap-x-6 gap-y-1 font-mono text-[11px] text-ink-2">
            <span>
              <span className="text-ink-3">features</span> {audit.n_features ?? "—"} distinct
            </span>
            <span>
              <span className="text-ink-3">seeds</span> {audit.seeds.join(", ") || "—"}
            </span>
            <span>
              <span className="text-ink-3">classes tested</span> {audit.classes.length}
            </span>
            <span>
              <span className="text-ink-3">rows per side</span> up to{" "}
              {count(audit.max_per_side)}
            </span>
          </div>
          <p className="text-[11px] text-ink-2 leading-relaxed max-w-prose">
            Run within one class so the classifier cannot win on class composition
            instead — 2017 is 83% Benign and 2018 88%, which would separate the corpora
            while saying nothing about whether the features mean the same thing. A class
            needs {count(audit.min_per_side)} rows on both sides before a balanced draw
            can say anything, which is why Infiltration is absent: the 2017 corpus holds
            36 such flows in total.
          </p>
        </div>
      </Panel>

      <AblationHeat audit={audit} />

      {audit.diagnostics ? (
        <>
          <LadderPanel audit={audit} />
          <SingleColumnPanel audit={audit} />
        </>
      ) : (
        <Panel className="p-4">
          <h3 className="text-[13px] font-semibold text-ink-0">
            This run was never stress-tested
          </h3>
          <p className="mt-2 text-[11px] text-ink-2 leading-relaxed max-w-prose">
            An AUC this high is the shape of a result that is usually an artefact. Run{" "}
            <code className="font-mono text-[10.5px]">python scripts/diagnose_av.py</code>{" "}
            to check it against a model-capacity ladder, a null control, and the
            duplicate and disjoint-column tests. Until then the number above is
            unverified — which is not the same as wrong.
          </p>
        </Panel>
      )}

      {audit.control ? (
        <>
          <ControlPanel control={audit.control} />
          {audit.control.tool_matched && (
            <ToolMatchedPanel matched={audit.control.tool_matched} />
          )}
        </>
      ) : (
        <Panel className="p-4">
          <h3 className="text-[13px] font-semibold text-ink-0">
            No within-dataset control
          </h3>
          <p className="mt-2 text-[11px] text-ink-2 leading-relaxed max-w-prose">
            The random-halves null above shows the procedure cannot invent separation,
            but a random split mixes every capture day into both halves — so it cannot
            rule out that the corpora differ the way any two capture sessions do. Run{" "}
            <code className="font-mono text-[10.5px]">
              python scripts/within_dataset_control.py
            </code>{" "}
            to draw the boundary inside one corpus instead.
          </p>
        </Panel>
      )}

      <RankingPanel audit={audit} diag={diagByClass} />
    </div>
  );
}

/**
 * The ablation curve as a grid: how much of the feature space has to go
 * before the two corpora stop being distinguishable.
 */
function AblationHeat({ audit }: { audit: AdversarialAudit }) {
  const ks = [0, ...audit.ablation_k];

  return (
    <Panel>
      <PanelHeader
        title="Separability as features are removed"
        sub="Each cell is the mean over the protocol's seeds. Column 0 is the full feature set."
      />
      <div className="px-4 py-3 space-y-3 overflow-x-auto">
        <table className="w-full min-w-[640px] border-separate border-spacing-1">
          <thead>
            <tr>
              <th className="w-28" />
              <th className="w-16" />
              {ks.map((k) => (
                <th
                  key={k}
                  scope="col"
                  className="text-[10px] font-mono text-ink-3 font-semibold pb-1"
                >
                  {k}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {audit.classes.map((c) => (
              <tr key={c.shared_class}>
                <th
                  scope="row"
                  className="text-right pr-2 text-[11px] text-ink-1 font-medium whitespace-nowrap"
                >
                  {c.shared_class}
                </th>
                <td className="text-right pr-2 text-[10px] font-mono text-ink-3 tabular-nums">
                  {count(c.n_per_side)}
                </td>
                {ks.map((k) => {
                  const v = k === 0 ? c.auc : c.ablation[String(k)];
                  return (
                    <td key={k}>
                      <div
                        title={`${c.shared_class}, ${k} removed: AUC ${score(v, 4)}`}
                        className={clsx(
                          "rounded border px-1 py-1.5 text-center font-mono text-[10.5px] tabular-nums",
                          TONE_CLASS[aucTone(v)],
                        )}
                      >
                        {isAbsent(v) ? <Nil /> : compactAuc(v as number)}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>

        <div className="flex flex-wrap items-center gap-3 text-[10px] font-mono text-ink-3">
          <span>features removed, of {audit.n_features ?? "—"} →</span>
          <span className="ml-auto flex items-center gap-2">
            <Swatch tone="chance" label="≈ chance" />
            <Swatch tone="mid" label="0.70" />
            <Swatch tone="high" label="0.90" />
            <Swatch tone="sep" label="0.99+" />
          </span>
        </div>

        <p className="text-[11px] text-ink-2 leading-relaxed max-w-prose">
          Separability barely responds until almost the entire feature space is gone. No
          small set of columns can be removed to take it away, which is what the ablation
          experiment then went on to confirm the hard way.
        </p>
      </div>
    </Panel>
  );
}

/**
 * Three characters wide so sixty cells fit across a phone.
 *
 * The leading zero is dropped because every value is a probability and the
 * column header already says so -- but only below 1.0, where dropping it
 * would turn a perfect score into ".000" and read as chance.
 */
function compactAuc(v: number): string {
  return v >= 0.9995 ? "1.00" : v.toFixed(3).slice(1);
}

function Swatch({ tone, label }: { tone: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <i className={clsx("h-2.5 w-5 rounded-sm border", TONE_CLASS[tone])} />
      {label}
    </span>
  );
}

/**
 * The capacity ladder with its null control — the check that decides whether
 * an AUC of 1.0 is signal or a model given enough rope to fit noise.
 */
function LadderPanel({ audit }: { audit: AdversarialAudit }) {
  const diag = audit.diagnostics!;
  const nulls = diag.classes.flatMap((c) =>
    Object.values(c.null_ladder).flatMap((rungs) => Object.values(rungs)),
  );
  const finite = nulls.filter((v): v is number => !isAbsent(v));

  return (
    <Panel>
      <PanelHeader
        title="The same rows and split, at five levels of model capacity"
        sub={diag.split ?? undefined}
        right={
          finite.length > 0 && (
            <Pill tone="ok" size="sm">
              null control {Math.min(...finite).toFixed(2)}–{Math.max(...finite).toFixed(2)}
            </Pill>
          )
        }
      />
      <div className="px-4 py-3 space-y-3 overflow-x-auto">
        <table className="w-full min-w-[620px] text-[11px]">
          <thead>
            <tr className="text-ink-3 text-[10px] uppercase tracking-[.14em]">
              <th className="text-left font-semibold pb-1.5">Class</th>
              {LADDER.map((r) => (
                <th
                  key={r.key}
                  title={r.hint}
                  className="text-right font-semibold pb-1.5 font-mono normal-case tracking-normal cursor-help"
                >
                  {r.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {diag.classes.map((c) => (
              <tr key={c.shared_class} className="border-t border-line-subtle">
                <td className="py-1.5 text-ink-0">{c.shared_class}</td>
                {LADDER.map((r) => {
                  const v = c.model_ladder[r.key];
                  const n = c.null_ladder.ids2017?.[r.key];
                  return (
                    <td
                      key={r.key}
                      title={`null control on 2017 halves: ${score(n, 4)}`}
                      className={clsx(
                        "py-1.5 text-right font-mono tabular-nums",
                        aucTone(v) === "sep" ? "text-danger" : "text-ink-1",
                      )}
                    >
                      {score(v, 4)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>

        <p className="text-[11px] text-ink-2 leading-relaxed max-w-prose">
          The null control is the identical procedure run on one corpus split into random
          halves, where there is nothing to find. Every rung lands at chance there, so
          none of them can manufacture separation. Read across a row: where the first
          rung is already near 1.0, a single threshold on a single column is enough, and
          a depth-1 tree has no capacity to memorise. Where the row climbs instead, the
          difference is spread thinly across the whole feature space.
        </p>
      </div>
    </Panel>
  );
}

/** One column, no model — and the near-duplicate and disjoint-range checks. */
function SingleColumnPanel({ audit }: { audit: AdversarialAudit }) {
  const diag = audit.diagnostics!;
  const chance = diag.near_dup_quantile;

  return (
    <Panel>
      <PanelHeader
        title="One column, no model"
        sub="Rank every row by a single column and ask how often a 2018 row outranks a 2017 one."
      />
      <div className="px-4 py-3 space-y-3 overflow-x-auto">
        <table className="w-full min-w-[720px] text-[11px]">
          <thead>
            <tr className="text-ink-3 text-[10px] uppercase tracking-[.14em]">
              <th className="text-left font-semibold pb-1.5">Class</th>
              <th className="text-left font-semibold pb-1.5">Most separating column</th>
              <th className="text-right font-semibold pb-1.5">Alone</th>
              <th className="text-right font-semibold pb-1.5">2017</th>
              <th className="text-right font-semibold pb-1.5">2018</th>
              <th
                className="text-right font-semibold pb-1.5 cursor-help"
                title="Share of test rows sitting closer to a training row than training rows sit to each other. The threshold is taken from the data, which makes the quantile itself the chance rate."
              >
                Near-dup
              </th>
              <th
                className="text-right font-semibold pb-1.5 cursor-help"
                title="Identical rows appearing on both sides of the train/test split. Anything above zero would mean the model can answer from memory."
              >
                Dup ↔ split
              </th>
            </tr>
          </thead>
          <tbody>
            {diag.classes.map((c) => {
              const disjoint =
                c.best_single_feature != null &&
                c.disjoint_columns.includes(c.best_single_feature);
              const elevated =
                !isAbsent(c.near_duplicate_test_share) &&
                !isAbsent(chance) &&
                (c.near_duplicate_test_share as number) > (chance as number) * 3;
              return (
                <tr key={c.shared_class} className="border-t border-line-subtle">
                  <td className="py-1.5 text-ink-0">{c.shared_class}</td>
                  <td className="py-1.5 font-mono text-[10.5px] text-ink-1">
                    {c.best_single_feature ?? <Nil />}
                    {disjoint && (
                      <span
                        className="ml-1.5 text-danger"
                        title="The 0.1st–99.9th percentile ranges of the two corpora do not overlap at all, so this class is separable by construction."
                      >
                        · disjoint
                      </span>
                    )}
                  </td>
                  <td
                    className={clsx(
                      "py-1.5 text-right font-mono tabular-nums",
                      aucTone(c.best_single_feature_auc) === "sep"
                        ? "text-danger"
                        : "text-ink-1",
                    )}
                  >
                    {score(c.best_single_feature_auc, 4)}
                  </td>
                  {["ids2017", "ids2018"].map((corpus) => (
                    <td
                      key={corpus}
                      title={`median in ${CORPUS_LABEL[corpus] ?? corpus}`}
                      className="py-1.5 text-right font-mono tabular-nums text-ink-2"
                    >
                      {isAbsent(c.best_single_feature_median[corpus]) ? (
                        <Nil />
                      ) : (
                        formatMedian(c.best_single_feature_median[corpus] as number)
                      )}
                    </td>
                  ))}
                  <td
                    className={clsx(
                      "py-1.5 text-right font-mono tabular-nums",
                      elevated ? "text-warn" : "text-ink-2",
                    )}
                  >
                    {percent(c.near_duplicate_test_share, 2)}
                  </td>
                  <td
                    className={clsx(
                      "py-1.5 text-right font-mono tabular-nums",
                      c.rows_identical_across_split ? "text-danger" : "text-ok",
                    )}
                  >
                    {count(c.rows_identical_across_split)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        <p className="text-[11px] text-ink-2 leading-relaxed max-w-prose">
          No model is involved in the <strong className="text-ink-1">Alone</strong>{" "}
          column. Where it reaches 0.99 the corpora are told apart by one number per
          flow. A near-duplicate share above{" "}
          {isAbsent(chance) ? "the chance rate" : percent(chance, 2)} — the rate chance
          alone produces, since the threshold is that quantile of the training data&apos;s
          own nearest-neighbour distances — means test rows sit unusually close to
          training rows; it inflates what a memorising model can do, but not what a
          single threshold can.{" "}
          {diag.preprocessing ? `Preprocessing: ${diag.preprocessing}.` : null}
        </p>
      </div>
    </Panel>
  );
}

/** Medians span decades between columns, so fixed decimals would be unreadable. */
function formatMedian(v: number): string {
  if (Number.isInteger(v)) return v.toLocaleString("en-US");
  if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0)) return v.toExponential(2);
  return v.toFixed(3);
}

/**
 * The within-dataset control.
 *
 * The random-halves null two panels up shows the procedure cannot invent
 * separation. It cannot speak to the claim, because shuffling rows mixes every
 * capture day into both halves — the rival explanation lives in exactly the
 * structure a random split destroys. This panel keeps that structure and asks
 * whether a boundary *inside* one corpus separates as well as the boundary
 * between them.
 *
 * Columns are ordered cross → within → null on purpose: left to right is
 * strongest boundary to no boundary at all, and the reader's question is where
 * along that row the AUC falls away.
 */
const CONTRAST_COLUMNS: { key: string; label: string; hint: string }[] = [
  {
    key: "cross",
    label: "2017 · 2018",
    hint: "The corpus boundary — the result everything else here is a control for.",
  },
  {
    key: "time_ids2017",
    label: "2017 early · late",
    hint: "Earliest half of this class against its latest half, by capture order. For a class confined to one capture file this splits a single attack window down the middle.",
  },
  {
    key: "time_ids2018",
    label: "2018 early · late",
    hint: "The same split inside CSE-CIC-IDS2018, by timestamp.",
  },
  {
    key: "group_ids2017",
    label: "2017 captures",
    hint: "One set of capture files against another, balanced by size rather than ordered by date. Absent for a class that lives in a single capture — in CICIDS2017 every attack class does.",
  },
  {
    key: "group_ids2018",
    label: "2018 captures",
    hint: "The same split across capture days in CSE-CIC-IDS2018.",
  },
  {
    key: "null_ids2017",
    label: "2017 random",
    hint: "Two random halves — a boundary known to mean nothing. Anything above chance here would invalidate the whole procedure.",
  },
  {
    key: "null_ids2018",
    label: "2018 random",
    hint: "The same null inside CSE-CIC-IDS2018.",
  },
];

const VERDICT_TONE: Record<string, "ok" | "warn" | "danger" | "muted"> = {
  "fingerprint confirmed": "ok",
  "fingerprint refuted": "danger",
  partial: "warn",
  inconclusive: "muted",
};

function ControlPanel({ control }: { control: WithinDatasetControl }) {
  const columns = useMemo(
    () =>
      CONTRAST_COLUMNS.filter((c) =>
        control.classes.some((k) => k.contrasts[c.key] !== undefined),
      ),
    [control.classes],
  );

  return (
    <Panel>
      <PanelHeader
        title="Is it the datasets, or just different capture sessions?"
        sub="Every cell in a row uses the same rows per side, so sample size cannot explain a difference between them."
        right={
          <Pill tone={VERDICT_TONE[control.verdict ?? ""] ?? "muted"} size="sm">
            {control.verdict ?? "not run"}
          </Pill>
        }
      />
      <div className="px-4 py-3 space-y-3 overflow-x-auto">
        <table className="w-full min-w-[760px] text-[11px]">
          <thead>
            <tr className="text-ink-3 text-[10px] uppercase tracking-[.14em]">
              <th className="text-left font-semibold pb-1.5">Class</th>
              <th className="text-right font-semibold pb-1.5">n / side</th>
              {columns.map((c) => (
                <th
                  key={c.key}
                  title={c.hint}
                  className="text-right font-semibold pb-1.5 cursor-help normal-case tracking-normal font-mono"
                >
                  {c.label}
                </th>
              ))}
              <th
                className="text-right font-semibold pb-1.5 cursor-help"
                title="Cross-dataset AUC minus the highest within-dataset contrast — the strongest case for the rival explanation, not the most convenient one."
              >
                Gap
              </th>
            </tr>
          </thead>
          <tbody>
            {control.classes.map((k) => (
              <tr key={k.shared_class} className="border-t border-line-subtle">
                <td className="py-1.5 text-ink-0">{k.shared_class}</td>
                <td className="py-1.5 text-right font-mono tabular-nums text-ink-3">
                  {count(k.n_per_side)}
                </td>
                {columns.map((c) => {
                  const cell = k.contrasts[c.key];
                  return (
                    <td key={c.key} className="py-1.5 text-right">
                      {cell === undefined ? (
                        <Nil reason="This class lives in a single capture, so there is no file boundary to draw." />
                      ) : (
                        <span
                          title={`${cell.boundary} · stump ${score(cell.stump, 4)} · ±${score(cell.auc_sd, 4)}`}
                          className={clsx(
                            "font-mono tabular-nums cursor-help",
                            cell.kind === "null"
                              ? "text-ink-3"
                              : aucTone(cell.auc) === "sep"
                                ? "text-danger"
                                : aucTone(cell.auc) === "high"
                                  ? "text-warn"
                                  : "text-ink-1",
                          )}
                        >
                          {score(cell.auc, 4)}
                        </span>
                      )}
                    </td>
                  );
                })}
                <td
                  className={clsx(
                    "py-1.5 text-right font-mono tabular-nums",
                    isAbsent(k.gap)
                      ? "text-ink-3"
                      : (k.gap as number) >= 0.1
                        ? "text-ok"
                        : "text-warn",
                  )}
                >
                  {isAbsent(k.gap) ? <Nil /> : `${(k.gap as number) > 0 ? "+" : ""}${(k.gap as number).toFixed(4)}`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {control.capture_day && <CaptureDayBlock capture={control.capture_day} />}

        <p className="text-[11px] text-ink-2 leading-relaxed max-w-prose">
          {control.reason ? `${control.reason[0].toUpperCase()}${control.reason.slice(1)}. ` : null}
          A capture file is a capture day in both corpora, so the two
          within-dataset contrasts differ in granularity rather than in kind; neither
          isolates &ldquo;file&rdquo; from &ldquo;date&rdquo;. What they answer together is whether any
          boundary inside a corpus separates as strongly as the boundary between
          corpora. The rule these numbers were judged against was written down before
          the run, in{" "}
          <code className="font-mono text-[10.5px]">CONTROL_PREREGISTRATION.md</code>.
        </p>
      </div>
    </Panel>
  );
}

/**
 * Why two of the rows above read 1.0000, and what that turned out to measure.
 *
 * The 2018 corpus gives each attack tool its own capture day, and the shared
 * seven-class schema collapses several tools into one class. So a contrast
 * labelled "one Tuesday against one Wednesday" can really be Hulk against
 * GoldenEye. Holding one of the two fixed at a time separates them, and the
 * gap between these two rows is the whole point of the block.
 */
function CaptureDayBlock({
  capture,
}: {
  capture: NonNullable<WithinDatasetControl["capture_day"]>;
}) {
  const groups = [
    {
      title: "Same attack tool, different capture day",
      sub: "The capture-session effect on its own",
      rows: capture.same_tool_other_day,
      tone: "text-ink-1",
    },
    {
      title: "Different attack tool, same collapsed class",
      sub: "The confound that made two rows above unreadable",
      rows: capture.other_tool_same_class,
      tone: "text-danger",
    },
  ];

  return (
    <div className="rounded border border-line-subtle bg-surface-elevated/40 p-3 space-y-3">
      {groups.map((g) => (
        <div key={g.title}>
          <div className="text-[9.5px] uppercase tracking-[.16em] text-ink-3 font-semibold">
            {g.title}
          </div>
          <div className="text-[10px] text-ink-3 mb-1.5">{g.sub}</div>
          {g.rows.length === 0 ? (
            <div className="text-[11px] text-ink-2">
              Nothing in the corpus can draw this contrast with at least{" "}
              {count(capture.min_per_side)} rows a side.
            </div>
          ) : (
            g.rows.map((r) => (
              <div
                key={`${r.label}-${r.boundary}`}
                className="flex items-baseline gap-3 text-[11px] py-0.5"
              >
                <span className={clsx("font-mono tabular-nums w-14", g.tone)}>
                  {score(r.auc, 4)}
                </span>
                <span className="text-ink-1 w-28 truncate">{r.label}</span>
                <span className="text-ink-3 text-[10px] font-mono truncate">
                  {r.boundary}
                </span>
                <span
                  className="ml-auto text-ink-3 text-[10px] font-mono"
                  title="A depth-1 stump on the same rows — it cannot memorise, so where it agrees the separation is real signal."
                >
                  stump {score(r.stump, 4)}
                </span>
              </div>
            ))
          )}
        </div>
      ))}
    </div>
  );
}

/**
 * The last thing that could have explained the result away.
 *
 * The shared seven-class schema puts several attack programs under one name,
 * and the capture-day follow-up showed two such programs separate at AUC 1.0.
 * That raises the same doubt about the headline: "DoS" names a different
 * mixture of tools in each corpus, so the classifier could have been
 * separating Hulk from GoldenEye and being credited with a dataset
 * fingerprint. Running the same program against itself across the two corpora
 * removes that possibility — or confirms it.
 *
 * Each row carries its class-level contrast at the same rows per side, so the
 * comparison is like-for-like rather than against a number from another draw.
 */
function ToolMatchedPanel({
  matched,
}: {
  matched: NonNullable<WithinDatasetControl["tool_matched"]>;
}) {
  const aucs = matched.tool_pairs
    .map((p) => p.same_tool.auc)
    .filter((v): v is number => !isAbsent(v));
  const lo = aucs.length ? Math.min(...aucs) : null;

  return (
    <Panel>
      <PanelHeader
        title="The same attack program, two labs, two years apart"
        sub="Hulk against Hulk. Only the corpus changes."
        right={
          <Pill tone={lo !== null && lo >= 0.99 ? "danger" : "warn"} size="sm">
            {lo === null ? "not run" : `lowest ${score(lo, 4)}`}
          </Pill>
        }
      />
      <div className="px-4 py-3 space-y-3 overflow-x-auto">
        <table className="w-full min-w-[680px] text-[11px]">
          <thead>
            <tr className="text-ink-3 text-[10px] uppercase tracking-[.14em]">
              <th className="text-left font-semibold pb-1.5">Tool</th>
              <th className="text-left font-semibold pb-1.5">Class</th>
              <th className="text-right font-semibold pb-1.5">n / side</th>
              <th
                className="text-right font-semibold pb-1.5 cursor-help"
                title="The same attack program on both sides. Only the lab and the year differ."
              >
                Same tool
              </th>
              <th
                className="text-right font-semibold pb-1.5 cursor-help"
                title="One threshold on one column, on the same rows. It cannot memorise."
              >
                stump
              </th>
              <th
                className="text-right font-semibold pb-1.5 cursor-help"
                title="The ordinary cross-dataset contrast for that class, re-run at the same rows per side."
              >
                Whole class
              </th>
            </tr>
          </thead>
          <tbody>
            {matched.tool_pairs.map((p) => (
              <tr key={p.tool} className="border-t border-line-subtle">
                <td className="py-1.5 text-ink-0">
                  {p.tool}
                  {p.identity_from_documentation && (
                    <span
                      className="ml-1.5 text-warn cursor-help"
                      title={`The two labels do not say the same word (${p.ids2017_label} / ${p.ids2018_label}); the pairing rests on CIC's documentation.`}
                    >
                      *
                    </span>
                  )}
                </td>
                <td className="py-1.5 text-ink-2">{p.shared_class}</td>
                <td className="py-1.5 text-right font-mono tabular-nums text-ink-3">
                  {count(p.n_per_side)}
                </td>
                <td
                  className={clsx(
                    "py-1.5 text-right font-mono tabular-nums",
                    aucTone(p.same_tool.auc) === "sep" ? "text-danger" : "text-ink-1",
                  )}
                >
                  {score(p.same_tool.auc, 4)}
                </td>
                <td className="py-1.5 text-right font-mono tabular-nums text-ink-2">
                  {score(p.same_tool.stump, 4)}
                </td>
                <td className="py-1.5 text-right font-mono tabular-nums text-ink-3">
                  {score(p.whole_class.auc, 4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {matched.different_tools_within_ids2017.length > 0 && (
          <div className="rounded border border-line-subtle bg-surface-elevated/40 p-3">
            <div className="text-[9.5px] uppercase tracking-[.16em] text-ink-3 font-semibold mb-1.5">
              Different tools inside CICIDS2017
            </div>
            {matched.different_tools_within_ids2017.map((r) => (
              <div
                key={r.shared_class}
                className="flex items-baseline gap-3 text-[11px] py-0.5"
              >
                <span className="font-mono tabular-nums w-14 text-danger">
                  {score(r.auc, 4)}
                </span>
                <span className="text-ink-1 w-28 truncate">{r.shared_class}</span>
                <span className="text-ink-3 text-[10px] font-mono truncate">
                  {r.tools.join(" / ")}
                </span>
                <span className="ml-auto text-ink-3 text-[10px] font-mono">
                  stump {score(r.stump, 4)}
                </span>
              </div>
            ))}
            <p className="mt-2 text-[10.5px] text-ink-2 leading-relaxed">
              The 2018 half of this reached 1.0000 twice. Agreement here makes
              collapsing tools into a class a property of the schema rather than a
              quirk of one corpus.
            </p>
          </div>
        )}

        <p className="text-[11px] text-ink-2 leading-relaxed max-w-prose">
          Holding the attack program fixed removes the one explanation the
          within-dataset control could not: that the corpora separate because
          &ldquo;DoS&rdquo; names a different mixture of tools on each side. Read the{" "}
          <strong className="text-ink-1">Same tool</strong> column against{" "}
          <strong className="text-ink-1">Whole class</strong> — where they agree, tool
          composition was never what the classifier was using. The stump column
          matters as much: a single threshold on a single column cannot memorise, so
          where it stays high the separation is a real property of the features rather
          than something a 300-tree ensemble assembled.
        </p>
      </div>
    </Panel>
  );
}

/** Which columns carry the separation, averaged over classes and seeds. */
function RankingPanel({
  audit,
  diag,
}: {
  audit: AdversarialAudit;
  diag: Map<string, AdversarialDiagnosticClass>;
}) {
  const [klass, setKlass] = useState<string | null>(null);

  const rows = useMemo(() => {
    if (klass === null) {
      return audit.ranking.slice(0, 15).map((r) => ({
        feature: r.feature,
        value: r.mean,
        extra: r.max,
      }));
    }
    const c = audit.classes.find((x) => x.shared_class === klass);
    return (c?.top_features ?? []).map((f) => ({
      feature: f.feature,
      value: f.importance,
      extra: f.sd,
    }));
  }, [audit, klass]);

  const peak = Math.max(
    ...rows.map((r) => (isAbsent(r.value) ? 0 : Math.abs(r.value as number))),
    1e-9,
  );
  const selected = klass ? diag.get(klass) : undefined;

  return (
    <Panel>
      <PanelHeader
        title="Which columns carry it"
        sub={
          klass === null
            ? "Mean permutation importance across every testable class"
            : `Permutation importance within ${klass}`
        }
        right={
          <div className="flex flex-wrap gap-1 justify-end">
            <Tab active={klass === null} onClick={() => setKlass(null)}>
              all classes
            </Tab>
            {audit.classes.map((c) => (
              <Tab
                key={c.shared_class}
                active={klass === c.shared_class}
                onClick={() => setKlass(c.shared_class)}
              >
                {c.shared_class}
              </Tab>
            ))}
          </div>
        }
      />
      <div className="px-4 py-3 space-y-2">
        {rows.length === 0 && (
          <p className="text-[11px] text-ink-2">This class recorded no ranking.</p>
        )}
        {rows.map((r) => (
          <div key={r.feature} className="grid grid-cols-[1fr_auto] gap-3 items-center">
            <div>
              <div className="flex justify-between text-[11px] mb-1 gap-3">
                <span className="font-mono text-[10.5px] text-ink-0 truncate">
                  {r.feature}
                </span>
                <span className="tabular-nums font-mono text-ink-1">
                  {score(r.value, 4)}
                </span>
              </div>
              <div className="h-1 rounded-full bg-surface-elevated overflow-hidden">
                <div
                  className="h-full rounded-full bg-info"
                  style={{
                    width: `${Math.min(100, (Math.abs((r.value as number) ?? 0) / peak) * 100)}%`,
                  }}
                />
              </div>
            </div>
            <span
              className="text-[10px] font-mono text-ink-3 tabular-nums w-16 text-right"
              title={klass === null ? "highest in any one class" : "spread across seeds"}
            >
              {klass === null ? `max ${score(r.extra, 3)}` : `±${score(r.extra, 3)}`}
            </span>
          </div>
        ))}

        <p className="pt-1 text-[11px] text-ink-2 leading-relaxed max-w-prose">
          Permutation importance measures what a column contributes that nothing else
          does, which is why a column can shift enormously between corpora and still rank
          last: the inter-arrival family moved by three to five decades and scores −0.0000,
          because it is heavily self-correlated and any one of its members can stand in
          for the rest. Duplicate columns were collapsed before this ran, or each would
          have hidden the other.
          {selected?.disjoint_columns.length ? (
            <>
              {" "}
              In {klass}, {selected.disjoint_columns.join(", ")} separates the corpora
              outright, with no overlap between their ranges.
            </>
          ) : null}
        </p>
      </div>
    </Panel>
  );
}

function Tab({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      className={clsx(
        "px-2 py-0.5 rounded-sm text-[10.5px] font-mono border transition-colors",
        active
          ? "border-line-base text-ink-0 bg-surface-elevated"
          : "border-transparent text-ink-2 hover:text-ink-0",
      )}
    >
      {children}
    </button>
  );
}
