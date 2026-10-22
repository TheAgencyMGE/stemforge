"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { InstrumentCatalog } from "@/lib/api";

const SEP: Record<string, string> = {
  isolated: "separates",
  shared: "inside another stem",
  masked: "mask only",
  unavailable: "no separator",
};

export function Coverage() {
  const [catalog, setCatalog] = useState<InstrumentCatalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.instruments().then(setCatalog).catch((e: unknown) => setError(e instanceof Error ? e.message : "Could not load the instrument catalog."));
  }, []);
  if (error) return <div className="faint">{error}</div>;
  if (!catalog) return <div className="faint">Loading instrument coverage…</div>;
  return (
    <div className="stack" style={{ gap: 10 }}>
      <p className="muted" style={{ margin: 0 }}>
        A name in this list is not a promise of its own model. “Separates” means an HTDemucs stem.
        Everything else stays mixed into the closest stem, or is only a listening mask.
      </p>
      {catalog.families.map((family) => (
        <details key={family.id} className="card" style={{ padding: "8px 12px" }}>
          <summary style={{ cursor: "pointer", textTransform: "capitalize" }}>{family.id}</summary>
          <div className="stack" style={{ gap: 8, marginTop: 8 }}>
            {family.instruments.map((inst) => (
              <div key={inst.id}>
                <div className="row" style={{ gap: 8 }}>
                  <strong style={{ fontSize: 13 }}>{inst.name}</strong>
                  <span className={`chip ${inst.separate === "isolated" ? "ok" : ""}`}>{SEP[inst.separate] ?? inst.separate}</span>
                  <span className="chip">{inst.transcribe}</span>
                  {inst.model && <span className="faint mono" style={{ fontSize: 11 }}>{inst.model}</span>}
                </div>
                <div className="faint" style={{ fontSize: 12.5 }}>{inst.limitation}</div>
              </div>
            ))}
          </div>
        </details>
      ))}
      <details className="card" style={{ padding: "8px 12px" }}>
        <summary style={{ cursor: "pointer" }}>Models reviewed and left out</summary>
        <div className="stack" style={{ gap: 8, marginTop: 8 }}>
          {catalog.excluded_models.map((m) => (
            <div key={m.id}>
              <strong className="mono" style={{ fontSize: 12.5 }}>{m.id}</strong>
              <div className="faint" style={{ fontSize: 12.5 }}>{m.task}. {m.reason}</div>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}
