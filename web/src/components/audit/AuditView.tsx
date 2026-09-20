"use client";

/**
 * Picks the view for whichever question an audit asked.
 *
 * The four audits share a layout name and nothing else, because they are four
 * different questions rather than four runs of one procedure. Discriminating
 * on `kind` here keeps that fact in one place: a page renders `AuditView` and
 * never learns which audits exist.
 *
 * The fallback is not decoration. A kind added to the registry before its view
 * exists would otherwise render as a blank page, which reads as "this audit
 * found nothing" — the opposite of what it means.
 */

import { Panel } from "@/components/ui/Panel";
import { AdversarialValidation } from "./AdversarialValidation";
import { FeatureMapping } from "./FeatureMapping";
import { StackingLeakage } from "./StackingLeakage";
import { Ablation } from "./Ablation";
import type { Audit } from "@/lib/bundles";

export function AuditView({ audit, id }: { audit: Audit; id: string }) {
  switch (audit.kind) {
    case "adversarial_validation":
      return <AdversarialValidation audit={audit} id={id} />;
    case "feature_mapping":
      return <FeatureMapping audit={audit} id={id} />;
    case "stacking_leakage":
      return <StackingLeakage audit={audit} id={id} />;
    case "ablation":
      return <Ablation audit={audit} id={id} />;
    default:
      return <Unknown kind={(audit as Audit).kind} id={id} />;
  }
}

function Unknown({ kind, id }: { kind: string; id: string }) {
  return (
    <Panel className="p-6">
      <h3 className="text-[13px] font-semibold text-ink-0">
        No view for this audit yet
      </h3>
      <p className="mt-2 text-[11.5px] text-ink-2 leading-relaxed max-w-prose">
        The API reports <code className="font-mono text-[10.5px]">{id}</code> as an audit
        of kind <code className="font-mono text-[10.5px]">{kind}</code>, which this
        dashboard does not know how to render. Its findings are on disk under{" "}
        <code className="font-mono text-[10.5px]">results/crossdataset/</code>. This is a
        missing view, not an empty result.
      </p>
    </Panel>
  );
}
