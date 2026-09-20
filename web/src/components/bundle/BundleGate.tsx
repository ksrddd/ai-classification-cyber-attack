"use client";

/**
 * Loads the selected bundle and hands it to a page, rendering the loading,
 * error and no-bundle states in one place.
 *
 * Every view needs the same fallbacks, and the mockup is explicit that
 * "no bundle" is a designed screen rather than an error page — it tells the
 * reader what to run instead of showing an empty chart.
 *
 * It also guards the layout mismatch. A cross-dataset bundle sits in the same
 * picker as the training runs but has no per-model metrics, no champion and
 * no single test split; a page written against `data.models` would render an
 * empty ranking, which reads as "these models scored nothing" rather than
 * "this question does not apply to this run". Pages that can show transfer
 * results pass a `crossdataset` renderer; every other page gets the explained
 * state below without having to know the layout exists. Putting the guard
 * here rather than in each page means a page added later cannot forget it.
 *
 * An audit bundle gets the same treatment for a sharper reason: it scores no
 * models at all. It asks whether a result can be believed -- whether the
 * columns mean the same thing on both sides, whether the ensemble can see its
 * own test set -- so there is no accuracy anywhere in it to draw.
 */

import { useEffect, useState } from "react";
import { Panel } from "@/components/ui/Panel";
import { useBundle } from "./BundleProvider";
import {
  type Audit,
  type BundleDetail,
  type CrossDataset,
  getBundle,
} from "@/lib/bundles";

export function BundleGate({
  children,
  crossdataset,
  audit,
  what,
}: {
  children: (data: BundleDetail) => React.ReactNode;
  /** Renderer for the cross-dataset layout. Omit if the page cannot show it. */
  crossdataset?: (cross: CrossDataset, data: BundleDetail) => React.ReactNode;
  /** Renderer for the audit layout. Omit if the page cannot show it. */
  audit?: (evidence: Audit, data: BundleDetail) => React.ReactNode;
  /** What this page would have shown, named in the empty state. */
  what: string;
}) {
  const { activeId, loading: listLoading, error: listError } = useBundle();
  const [data, setData] = useState<BundleDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!activeId) return;
    let cancelled = false;
    setData(null);
    setError(null);
    getBundle(activeId)
      .then((d) => !cancelled && setData(d))
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [activeId]);

  // Order matters. "We could not reach the API" and "you have not trained
  // anything" are different problems with different fixes, and showing the
  // second when the first is true sends the reader to retrain a model that
  // already exists.
  if (listError) {
    return (
      <Empty title={`${what} could not load`}>
        The dashboard could not reach the API at{" "}
        <Code>{process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}</Code> —{" "}
        {listError}. Start it with{" "}
        <Code>python -m uvicorn api.main:app --port 8000</Code>. This is a connection
        problem, not a missing result: any bundles on disk are still there.
      </Empty>
    );
  }

  if (error) {
    return (
      <Empty title={`${what} is unavailable`}>
        The API answered, but this bundle could not be read: {error}
      </Empty>
    );
  }

  if (listLoading) {
    return <Loading />;
  }

  if (!activeId) {
    return (
      <Empty title={`No results bundle behind ${what}`}>
        The API is reachable and reports nothing under <Code>results/</Code> that looks
        like a bundle. Train a run with <Code>python main.py --stage train</Code> for
        CICIDS2017, or <Code>python -m src.ids2018.train_ids2018</Code> for
        CSE-CIC-IDS2018, then pick it in the rail.
      </Empty>
    );
  }

  if (!data) return <Loading />;

  if (data.layout === "crossdataset") {
    const cross = data.crossdataset;
    if (!cross) {
      return (
        <Empty title={`${what} is unavailable`}>
          This bundle reports the cross-dataset layout but carries no transfer
          payload, which means its summary tables were never written. Run{" "}
          <Code>python scripts/analyze_crossdataset.py</Code> over{" "}
          <Code>{data.id}</Code>.
        </Empty>
      );
    }
    if (!crossdataset) {
      return (
        <Empty title={`${what} does not apply to a cross-dataset run`}>
          <Code>{data.id}</Code> is not a training run. It measures how much of what
          a model learned on one corpus survives being tested on the other, so its
          unit of result is a pair of corpora rather than a model — there is no
          single score per model to rank here, and no champion to name. Open{" "}
          <strong className="text-ink-1">Overview</strong> for the transfer matrix,
          or pick a training bundle in the rail to see {what}.
        </Empty>
      );
    }
    return <>{crossdataset(cross, data)}</>;
  }

  if (data.layout === "audit") {
    const evidence = data.audit;
    if (!evidence) {
      return (
        <Empty title={`${what} is unavailable`}>
          This bundle reports the audit layout but carries no findings, which means the
          audit script wrote its directory and then failed. Re-run it and check{" "}
          <Code>results/crossdataset/{data.run.run_name ?? data.id}</Code>.
        </Empty>
      );
    }
    if (!audit) {
      return (
        <Empty title={`${what} does not apply to an audit`}>
          <Code>{data.id}</Code> trained nothing. It is a check on whether an existing
          result can be believed, so it has no models to rank, no test split and no
          accuracy to report. Open <strong className="text-ink-1">Overview</strong> for
          its findings, or pick a training bundle in the rail to see {what}.
        </Empty>
      );
    }
    return <>{audit(evidence, data)}</>;
  }

  return <>{children(data)}</>;
}

function Loading() {
  return (
    <div className="space-y-2" aria-busy="true" aria-live="polite">
      <span className="sr-only">Loading results</span>
      {[...Array(6)].map((_, i) => (
        <div key={i} className="h-10 rounded-sm bg-surface-raised animate-pulse" />
      ))}
    </div>
  );
}

export function Empty({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Panel className="p-6">
      <h3 className="text-[13px] font-semibold text-ink-0">{title}</h3>
      <p className="mt-2 text-[11.5px] text-ink-2 leading-relaxed max-w-prose">{children}</p>
    </Panel>
  );
}

export function Code({ children }: { children: React.ReactNode }) {
  return (
    <code className="font-mono text-[10.5px] py-0.5 px-[5px] rounded bg-surface-elevated ring-1 ring-line-base text-ink-1">
      {children}
    </code>
  );
}
