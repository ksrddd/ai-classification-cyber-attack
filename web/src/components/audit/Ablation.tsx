"use client";

/**
 * Does dropping the columns that name the corpus improve transfer?
 *
 * The honest way to show a negative result is to show the rule it was judged
 * against *first*, which is why the pre-registration sits at the top of this
 * view rather than in a footnote. With 60 features, five seeds and two split
 * modes, something will always look like it moved; a threshold written down
 * afterwards is not a threshold.
 *
 * Per-model recovery is laid out wide — one column per K — so a row can be
 * read across. A model that falls monotonically as more columns are dropped is
 * telling a different story from one that dips and recovers, and melting the
 * table into long form would take that away.
 */

import { useMemo } from "react";
import { clsx } from "clsx";
import { Panel, PanelHeader } from "@/components/ui/Panel";
import { Pill } from "@/components/ui/Pill";
import { Nil } from "@/components/ui/Nil";
import { modelColor } from "@/lib/colors";
import { type AblationAudit, isAbsent, score } from "@/lib/bundles";

/** Always signed: "5.0 to -20.1" hides that one of the two is an improvement. */
function signed(v: number | null): string {
  return v === null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(1)}`;
}

export function Ablation({ audit, id }: { audit: AblationAudit; id: string }) {
  const modes = useMemo(
    () =>
      audit.summary
        .map((r) => r.mode)
        .filter((m, i, all) => all.indexOf(m) === i)
        .sort((a, b) => (a === "chronological" ? -1 : b === "chronological" ? 1 : 0)),
    [audit.summary],
  );

  const changes = audit.per_model
    .map((m) => m.change)
    .filter((v): v is number => !isAbsent(v));
  const worst = changes.length ? Math.min(...changes) : null;
  const best = changes.length ? Math.max(...changes) : null;

  return (
    <div className="space-y-4">
      <Panel>
        <PanelHeader
          eyebrow={id}
          title="Does removing the fingerprint help?"
          sub="Drop the top-K columns by dataset-separability importance, then re-run the whole transfer protocol."
          right={
            <Pill tone={best !== null && best > 0 ? "warn" : "danger"} size="sm">
              {best === null
                ? "no result"
                : `${signed(worst)} to ${signed(best)} pts across models`}
            </Pill>
          }
        />
        <div className="px-4 py-3">
          <p className="text-[11px] text-ink-2 leading-relaxed max-w-prose">
            Recovered percent is the share of the distance from the majority-class
            baseline to the ceiling that a transfer closes, which is what makes models
            with different ceilings comparable. If the dropped columns were pure capture
            artefact, removing them should narrow the gap. Watch whether the ceiling
            falls alongside the transfer instead — that is the signature of removing
            columns the attack itself needed.
          </p>
        </div>
      </Panel>

      {audit.preregistration && <Prereg text={audit.preregistration} />}

      <Panel>
        <PanelHeader
          title="Transfer as features are dropped"
          sub="Averaged over models and both directions, per split mode."
        />
        <div className="px-4 py-3 overflow-x-auto">
          <table className="w-full min-w-[560px] text-[11px]">
            <thead>
              <tr className="text-ink-3 text-[10px] uppercase tracking-[.14em]">
                <th className="text-left font-semibold pb-1.5">Split</th>
                <th className="text-right font-semibold pb-1.5">Dropped</th>
                <th className="text-right font-semibold pb-1.5">Ceiling</th>
                <th className="text-right font-semibold pb-1.5">Transfer</th>
                <th className="text-right font-semibold pb-1.5">Gap</th>
                <th className="text-right font-semibold pb-1.5">Recovered</th>
              </tr>
            </thead>
            <tbody>
              {modes.map((mode) =>
                audit.summary
                  .filter((r) => r.mode === mode)
                  .sort((a, b) => (a.k ?? 0) - (b.k ?? 0))
                  .map((r, i) => (
                    <tr key={`${mode}-${r.k}`} className="border-t border-line-subtle">
                      <td className="py-1.5 text-ink-2">{i === 0 ? mode : ""}</td>
                      <td className="py-1.5 text-right font-mono tabular-nums text-ink-1">
                        {r.k ?? 0}
                      </td>
                      <td className="py-1.5 text-right font-mono tabular-nums text-ink-2">
                        {score(r.ceiling, 4)}
                      </td>
                      <td className="py-1.5 text-right font-mono tabular-nums text-ink-1">
                        {score(r.transfer, 4)}
                      </td>
                      <td className="py-1.5 text-right font-mono tabular-nums text-ink-2">
                        {score(r.gap, 4)}
                      </td>
                      <td
                        className={clsx(
                          "py-1.5 text-right font-mono tabular-nums",
                          isAbsent(r.recovered_pct)
                            ? "text-ink-3"
                            : (r.recovered_pct as number) >= 25
                              ? "text-ink-0"
                              : "text-warn",
                        )}
                      >
                        {isAbsent(r.recovered_pct) ? (
                          <Nil />
                        ) : (
                          `${(r.recovered_pct as number).toFixed(1)}%`
                        )}
                      </td>
                    </tr>
                  )),
              )}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel>
        <PanelHeader
          title="Every model, every K"
          sub="Recovered percent, chronological split. Read across a row."
        />
        <div className="px-4 py-3 overflow-x-auto">
          <table className="w-full min-w-[520px] text-[11px]">
            <thead>
              <tr className="text-ink-3 text-[10px] uppercase tracking-[.14em]">
                <th className="text-left font-semibold pb-1.5">Model</th>
                {audit.ks.map((k) => (
                  <th key={k} className="text-right font-semibold pb-1.5 font-mono">
                    {k === 0 ? "none" : `−${k}`}
                  </th>
                ))}
                <th className="text-right font-semibold pb-1.5">Change</th>
              </tr>
            </thead>
            <tbody>
              {audit.per_model.map((m) => (
                <tr key={m.model} className="border-t border-line-subtle">
                  <td className="py-1.5 text-ink-0 flex items-center gap-2">
                    <span
                      className="h-2 w-2 rounded-full flex-shrink-0"
                      style={{ background: modelColor(m.model) }}
                    />
                    {m.model}
                  </td>
                  {audit.ks.map((k) => (
                    <td
                      key={k}
                      className="py-1.5 text-right font-mono tabular-nums text-ink-1"
                    >
                      {isAbsent(m.recovered_pct[String(k)]) ? (
                        <Nil />
                      ) : (
                        `${(m.recovered_pct[String(k)] as number).toFixed(1)}`
                      )}
                    </td>
                  ))}
                  <td
                    className={clsx(
                      "py-1.5 text-right font-mono tabular-nums",
                      isAbsent(m.change)
                        ? "text-ink-3"
                        : (m.change as number) > 0
                          ? "text-ok"
                          : "text-danger",
                    )}
                  >
                    {isAbsent(m.change) ? (
                      <Nil />
                    ) : (
                      `${(m.change as number) > 0 ? "+" : ""}${(m.change as number).toFixed(1)}`
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="pt-3 text-[11px] text-ink-2 leading-relaxed max-w-prose">
            Change is the last K minus none. Where every model moves the same way the
            result is about the feature set; where the tree ensembles lose and the linear
            and neural models do not, model family and feature set are interacting, and
            a single averaged number would have hidden that.
          </p>
        </div>
      </Panel>
    </div>
  );
}

/**
 * The pre-registered rule, rendered as the plain text it was written as.
 *
 * Deliberately not parsed into styled markdown: this is a record of what was
 * committed before the runs existed, and reformatting it invites editing it.
 */
function Prereg({ text }: { text: string }) {
  return (
    <Panel>
      <PanelHeader
        title="The rule, written down first"
        sub="Recorded before any of these runs existed"
        right={
          <Pill tone="info" size="sm">
            pre-registered
          </Pill>
        }
      />
      <pre className="px-4 py-3 text-[10.5px] font-mono text-ink-1 leading-relaxed whitespace-pre-wrap max-h-80 overflow-y-auto">
        {text}
      </pre>
    </Panel>
  );
}
