"use client";

/**
 * Loads the selected bundle and hands it to a page, rendering the loading,
 * error and no-bundle states in one place.
 *
 * Every view needs the same three fallbacks, and the mockup is explicit that
 * "no bundle" is a designed screen rather than an error page — it tells the
 * reader what to run instead of showing an empty chart.
 */

import { useEffect, useState } from "react";
import { Panel } from "@/components/ui/Panel";
import { useBundle } from "./BundleProvider";
import { type BundleDetail, getBundle } from "@/lib/bundles";

export function BundleGate({
  children,
  what,
}: {
  children: (data: BundleDetail) => React.ReactNode;
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
