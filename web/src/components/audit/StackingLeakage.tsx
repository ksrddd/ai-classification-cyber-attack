"use client";

/**
 * Can the stacking ensemble see the test set it is scored on?
 *
 * Reading the code says it cannot. Code reading is not evidence, so each
 * corpus gets the full battery and the results are shown per corpus rather
 * than merged. Three of the checks are protocol-level and cannot differ
 * between datasets; the duplicate-flow one can, because the raw 2018 corpus
 * is 29% duplicate flows and is therefore the side where getting the order of
 * deduplication and splitting wrong would do real damage. A merged verdict
 * would hide which corpus actually proved it.
 *
 * A check that cannot fail proves nothing, so the detail string for each one
 * is shown rather than a tick: "0 shared position(s)" and "coefficient
 * distance: out-of-fold 0.000000, in-sample 0.052408" are the evidence, and
 * the tick is only a summary of them.
 */

import { useMemo } from "react";
import { clsx } from "clsx";
import { Check, X } from "lucide-react";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { Pill } from "@/components/ui/Pill";
import { type StackingLeakageAudit, count } from "@/lib/bundles";

type Corpus = StackingLeakageAudit["corpora"][number];
type LeakCheck = Corpus["checks"][number];

export function StackingLeakage({
  audit,
  id,
}: {
  audit: StackingLeakageAudit;
  id: string;
}) {
  const totals = useMemo(
    () =>
      audit.corpora.reduce(
        (acc, c) => ({
          checks: acc.checks + (c.n_checks ?? 0),
          failed: acc.failed + (c.n_failed ?? 0),
        }),
        { checks: 0, failed: 0 },
      ),
    [audit.corpora],
  );

  return (
    <div className="space-y-4">
      <Panel>
        <PanelHeader
          eyebrow={id}
          title="Can the stacking ensemble see the test set?"
          sub="Run separately on each corpus. Every check below is one that could have failed."
          right={
            <Pill tone={totals.failed ? "danger" : "ok"} size="sm">
              {totals.checks - totals.failed} / {totals.checks} pass
            </Pill>
          }
        />
        <div className="px-4 py-3">
          <p className="text-[11px] text-ink-2 leading-relaxed max-w-prose">
            Stacking was the model whose transfer score moved most when the corpora were
            rebuilt, which makes it the one worth testing rather than the one worth
            trusting. The decisive check is the third: both candidate meta-feature
            matrices are rebuilt from scratch — one out-of-fold, one in-sample — a fresh
            meta-learner is fitted on each, and the real ensemble&apos;s coefficients are
            compared against both. Reproducing the out-of-fold version exactly is
            something an ensemble fitted on its own training predictions could not do.
          </p>
        </div>
      </Panel>

      <div className="grid lg:grid-cols-2 gap-4">
        {audit.corpora.map((corpus) => (
          <CorpusPanel key={corpus.dataset} corpus={corpus} />
        ))}
      </div>
    </div>
  );
}

function CorpusPanel({ corpus }: { corpus: Corpus }) {
  // Sections come from the audit script already ordered and prefixed ("1. ",
  // "2. "), so grouping by first appearance preserves that order without
  // this file having to know what the sections are.
  const sections = useMemo(() => {
    const out: [string, LeakCheck[]][] = [];
    corpus.checks.forEach((c) => {
      const found = out.find(([name]) => name === c.section);
      if (found) found[1].push(c);
      else out.push([c.section, [c]]);
    });
    return out;
  }, [corpus.checks]);

  return (
    <Panel>
      <PanelHeader
        title={corpus.label}
        sub={`${count(corpus.audit_rows)} audit rows · seed ${corpus.seed ?? "—"}`}
        right={
          <Pill tone={corpus.passed ? "ok" : "danger"} size="sm">
            {corpus.passed ? "no leakage" : `${corpus.n_failed} failed`}
          </Pill>
        }
      />
      <div className="px-4 py-3 space-y-3">
        {sections.map(([section, checks]) => (
          <div key={section}>
            <div className="text-[9.5px] uppercase tracking-[.16em] text-ink-3 font-semibold mb-1.5">
              {section}
            </div>
            <div className="space-y-1.5">
              {checks.map((c: LeakCheck) => (
                <div key={c.check} className="flex gap-2 items-start">
                  <span
                    className={clsx(
                      "mt-[3px] flex-shrink-0 grid place-items-center h-3.5 w-3.5 rounded-sm",
                      c.passed ? "bg-ok/15 text-ok" : "bg-danger/15 text-danger",
                    )}
                  >
                    {c.passed ? <Check size={10} strokeWidth={3} /> : <X size={10} strokeWidth={3} />}
                  </span>
                  <div className="min-w-0">
                    <div className="text-[11px] text-ink-0 leading-snug">{c.check}</div>
                    <div className="text-[10px] font-mono text-ink-2 leading-snug break-words">
                      {c.detail}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
        {corpus.run_utc && (
          <div className="pt-1 text-[10px] font-mono text-ink-3">run {corpus.run_utc}</div>
        )}
      </div>
    </Panel>
  );
}
