"use client";

/**
 * Does column 12 of one corpus mean the same thing as column 12 of the other?
 *
 * The schema paired them by position in CICFlowMeter's emission order and the
 * pairs were confirmed by reading them back, which is a visual check. This is
 * the one that can fail: every identity is algebraic, and each is evaluated on
 * each corpus *separately*, so a pass never depends on the two corpora
 * agreeing about anything. That independence is the whole point — a
 * distribution comparison cannot tell a mis-mapped column from a column that
 * genuinely shifted, and mistaking the first for the second would have put our
 * own bug at the top of the fingerprint ranking.
 */

import { useState } from "react";
import { clsx } from "clsx";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { Pill } from "@/components/ui/Pill";
import { Nil } from "@/components/ui/Nil";
import { type FeatureMappingAudit, count, isAbsent, percent } from "@/lib/bundles";

const CORPUS_LABEL: Record<string, string> = {
  ids2017: "CICIDS2017",
  ids2018: "CSE-CIC-IDS2018",
};

export function FeatureMapping({
  audit,
  id,
}: {
  audit: FeatureMappingAudit;
  id: string;
}) {
  const failed = audit.invariants.filter((i) => i.verdict !== "PASS");
  const flagged = audit.scale_audit.filter((s) => s.flag !== "OK");

  return (
    <div className="space-y-4">
      <Panel>
        <PanelHeader
          eyebrow={id}
          title="Do the 77 features mean the same thing?"
          sub="Algebraic identities, evaluated on each corpus separately. A pass does not assume the corpora agree."
          right={
            <Pill tone={failed.length ? "danger" : "ok"} size="sm">
              {audit.invariants.length - failed.length} / {audit.invariants.length}{" "}
              invariants hold
            </Pill>
          }
        />
        <div className="px-4 py-3 space-y-3">
          <div className="flex flex-wrap gap-x-6 gap-y-1 font-mono text-[11px] text-ink-2">
            {Object.entries(audit.n_rows).map(([corpus, n]) => (
              <span key={corpus}>
                <span className="text-ink-3">{CORPUS_LABEL[corpus] ?? corpus}</span>{" "}
                {count(n)} rows
              </span>
            ))}
            <span>
              <span className="text-ink-3">columns</span> {audit.n_features ?? "—"}
            </span>
            <span>
              <span className="text-ink-3">distinct measurements</span>{" "}
              {audit.n_distinct_measurements ?? "—"}
            </span>
          </div>
          <p className="text-[11px] text-ink-2 leading-relaxed max-w-prose">
            Each identity is a statement that must be true of a single corpus on its own
            — mean forward packet length <em>is</em> total bytes over packet count, and
            nothing about the other dataset enters that claim. An identity that holds on
            one side and fails on the other therefore localises the error to the columns
            it touches, which is what makes this decisive where a distribution comparison
            is not.
          </p>
        </div>
      </Panel>

      <Panel>
        <PanelHeader
          title="The identities"
          sub="Violation rate on each corpus. Zero on both is the only passing answer."
        />
        <div className="px-4 py-3 overflow-x-auto">
          <table className="w-full min-w-[640px] text-[11px]">
            <thead>
              <tr className="text-ink-3 text-[10px] uppercase tracking-[.14em]">
                <th className="text-left font-semibold pb-1.5">Identity</th>
                <th className="text-right font-semibold pb-1.5">2017</th>
                <th className="text-right font-semibold pb-1.5">2018</th>
                <th className="text-left font-semibold pb-1.5 pl-4">What it settles</th>
              </tr>
            </thead>
            <tbody>
              {audit.invariants.map((inv) => (
                <tr key={inv.invariant} className="border-t border-line-subtle align-top">
                  <td className="py-1.5 pr-3">
                    <span className="font-mono text-[10.5px] text-ink-0">
                      {inv.invariant}
                    </span>
                    <div className="text-[10px] text-ink-3 font-mono mt-0.5">
                      {inv.columns.join(" · ")}
                    </div>
                  </td>
                  <Rate value={inv.rate_ids2017} />
                  <Rate value={inv.rate_ids2018} />
                  <td className="py-1.5 pl-4 text-ink-2 max-w-[38ch]">{inv.note ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <div className="grid lg:grid-cols-2 gap-4">
        <Panel>
          <PanelHeader
            title="Duplicate measurement groups"
            sub={`${audit.n_features ?? "—"} columns hold ${audit.n_distinct_measurements ?? "—"} distinct measurements`}
          />
          <div className="px-4 py-3 space-y-1.5">
            {audit.duplicate_groups.map((group) => (
              <div
                key={group.join("|")}
                className="font-mono text-[10.5px] text-ink-1 leading-relaxed"
              >
                {group.length > 3 ? (
                  <>
                    {group.slice(0, 2).join(" = ")}{" "}
                    <span className="text-ink-3">
                      + {group.length - 2} more all-zero column
                      {group.length - 2 === 1 ? "" : "s"}
                    </span>
                  </>
                ) : (
                  group.join(" = ")
                )}
              </div>
            ))}
            <p className="pt-2 text-[11px] text-ink-2 leading-relaxed">
              This is not bookkeeping. Permutation importance splits credit between
              duplicates — shuffle one and the model reads the other, so both score near
              zero. Ranking the fingerprint without collapsing these first would have
              hidden exactly the columns that search exists to find.
            </p>
          </div>
        </Panel>

        <ScalePanel flagged={flagged} total={audit.scale_audit.length} />
      </div>
    </div>
  );
}

function Rate({ value }: { value: number | null }) {
  const bad = !isAbsent(value) && (value as number) > 0;
  return (
    <td
      className={clsx(
        "py-1.5 text-right font-mono tabular-nums",
        bad ? "text-danger" : "text-ok",
      )}
    >
      {isAbsent(value) ? <Nil /> : percent(value, 2)}
    </td>
  );
}

/**
 * Reported beside the invariants and deliberately not styled as a verdict: a
 * column can shift by decades between corpora and still be the same
 * measurement. That is the distinction the invariants draw, and this table
 * would undo it if it looked like a failure list.
 */
function ScalePanel({
  flagged,
  total,
}: {
  flagged: FeatureMappingAudit["scale_audit"];
  total: number;
}) {
  const [open, setOpen] = useState(false);
  const shown = open ? flagged : flagged.slice(0, 8);

  return (
    <Panel>
      <PanelHeader
        title="Where the two corpora sit differently"
        sub={`${flagged.length} of ${total} columns flagged — an observation, not a failure`}
      />
      <div className="px-4 py-3 space-y-2">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="text-ink-3 text-[10px] uppercase tracking-[.14em]">
              <th className="text-left font-semibold pb-1.5">Column</th>
              <th className="text-right font-semibold pb-1.5">Median 2017</th>
              <th className="text-right font-semibold pb-1.5">Median 2018</th>
              <th className="text-left font-semibold pb-1.5 pl-3">Flag</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((s) => (
              <tr key={s.feature} className="border-t border-line-subtle">
                <td className="py-1.5 font-mono text-[10.5px] text-ink-0">{s.feature}</td>
                <Median value={s.ids2017.median} />
                <Median value={s.ids2018.median} />
                <td className="py-1.5 pl-3 text-[10px] font-mono text-ink-3" title={s.reason}>
                  {s.flag}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {flagged.length > shown.length && (
          <button
            onClick={() => setOpen(true)}
            className="text-[10.5px] font-mono text-info hover:underline"
          >
            show {flagged.length - shown.length} more
          </button>
        )}

        <p className="pt-1 text-[11px] text-ink-2 leading-relaxed">
          A shift here is a fact about the traffic or the capture, not about the schema.
          The identities above already proved these columns are the same measurement; how
          far apart their distributions sit is the question the adversarial validation
          takes up.
        </p>
      </div>
    </Panel>
  );
}

function Median({ value }: { value: number | null }) {
  return (
    <td className="py-1.5 text-right font-mono tabular-nums text-ink-1">
      {isAbsent(value) ? (
        <Nil />
      ) : Number.isInteger(value) ? (
        (value as number).toLocaleString("en-US")
      ) : Math.abs(value as number) >= 1e4 || Math.abs(value as number) < 1e-2 ? (
        (value as number).toExponential(1)
      ) : (
        (value as number).toFixed(2)
      )}
    </td>
  );
}
