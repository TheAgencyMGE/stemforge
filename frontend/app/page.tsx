"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { AppHeader } from "@/components/Brand";
import { SampleButton, UploadPanel } from "@/components/UploadPanel";
import { api, errorMessage, importBundle } from "@/lib/api";
import { fmtAgo, fmtTime } from "@/lib/format";
import { useEditor } from "@/lib/store";
import type { ProjectSummary, SystemInfo } from "@/lib/types";

const STATUS_CHIP: Record<string, string> = { ready: "ok", processing: "accent", queued: "accent", failed: "err", cancelled: "warn" };

export default function Home() {
  const router = useRouter();
  const toast = useEditor((s) => s.toast);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [systemErr, setSystemErr] = useState<string | null>(null);
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const importRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      setProjects(await api.listProjects());
    } catch (e) {
      setSystemErr(errorMessage(e));
    }
  }, []);

  useEffect(() => {
    api.system().then(setSystem).catch((e) => setSystemErr(errorMessage(e)));
    // Fetch-on-mount + polling: state is set from an external system (the API).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [refresh]);

  const missingModel = system?.models.find((m) => m.name === "htdemucs" && !m.installed);

  return (
    <>
      <AppHeader>
        <input
          ref={importRef}
          type="file"
          accept=".zip,application/zip"
          className="sr-only"
          aria-label="Open a project bundle"
          onChange={async (e) => {
            const f = e.target.files?.[0];
            e.target.value = "";
            if (!f) return;
            try {
              const p = await importBundle(f);
              toast("success", `Restored “${p.name}”`);
              router.push(`/projects/${p.id}`);
            } catch (err) {
              toast("error", `Could not open bundle: ${errorMessage(err)}`);
            }
          }}
        />
        <button className="btn ghost" onClick={() => importRef.current?.click()}>
          Open project bundle…
        </button>
      </AppHeader>
      <main className="home">
        {systemErr && (
          <div className="errbox" role="alert">
            Can&apos;t reach the Stemforge API ({systemErr}). Start it with <code>stemforge-api</code> and a worker with{" "}
            <code>stemforge-worker</code>; see README.
          </div>
        )}
        {missingModel && (
          <div className="warnbox" role="alert">
            The separation model isn&apos;t installed yet. Run <code>stemforge-models download</code> on the server (≈140 MB, SHA-256
            verified).
          </div>
        )}
        <section className="hero">
          <div className="hero-copy card" style={{ padding: 22, display: "grid", alignContent: "space-between", gap: 18 }}>
            <div>
              <h1>Take any song apart</h1>
              <p className="muted" style={{ margin: 0, maxWidth: 520 }}>
                Stemforge separates a mix into vocals, drums, bass and the rest, transcribes each stem to MIDI with an engine
                picked for that instrument, and gives you a synced editor to fix the notes before you export.
              </p>
              <div className="pipeline-steps" aria-label="Workflow">
                {["Upload", "→", "Analyze", "→", "Separate", "→", "Transcribe", "→", "Edit", "→", "Export"].map((s, i) => (
                  <span key={i} style={s === "→" ? { background: "none", border: 0, padding: 0, color: "var(--text-3)" } : undefined}>
                    {s}
                  </span>
                ))}
              </div>
            </div>
            <div className="row wrap" style={{ gap: 12 }}>
              <SampleButton />
              <span className="faint" style={{ fontSize: 12.5, maxWidth: 360 }}>
                A 30-second CC0 song with known ground-truth notes. It runs through the real pipeline.
              </span>
            </div>
          </div>
          <UploadPanel system={system} />
        </section>

        <section className="stack">
          <div className="row">
            <h2>Projects</h2>
            <span className="spacer" />
            {system && system.retention_days > 0 && (
              <span className="faint" style={{ fontSize: 12 }}>Uploads are deleted automatically after {system.retention_days} days.</span>
            )}
          </div>
          {projects === null ? (
            <div className="faint">Loading…</div>
          ) : projects.length === 0 ? (
            <div className="card empty">
              <div style={{ fontSize: 15, color: "var(--text)" }}>No projects yet</div>
              <div>Upload a song above, or try the demo to see the whole workflow in about a minute.</div>
            </div>
          ) : (
            <div className="project-grid">
              {projects.map((p) => (
                <Link key={p.id} href={`/projects/${p.id}`} className="card project-card">
                  <div className="row">
                    <span className="title">{p.name}</span>
                    <span className="spacer" />
                    <span className={`chip ${STATUS_CHIP[p.status] ?? ""}`}>{p.status}</span>
                  </div>
                  <div className="faint num" style={{ fontSize: 12 }}>
                    {p.duration ? fmtTime(p.duration, 0) : "–"} · {p.tempo ? `${p.tempo.toFixed(1)} BPM` : "tempo –"} · {p.key ?? "key –"} ·{" "}
                    {p.stems} stems
                  </div>
                  <div className="row faint" style={{ fontSize: 12 }}>
                    <span>{fmtAgo(p.created_at)}</span>
                    <span className="spacer" />
                    <button
                      className="btn ghost sm danger"
                      aria-label={`Delete ${p.name}`}
                      onClick={async (e) => {
                        e.preventDefault();
                        if (!confirm(`Delete “${p.name}” and all of its stems and MIDI? This can't be undone.`)) return;
                        try {
                          await api.deleteProject(p.id);
                          toast("info", "Project deleted");
                          void refresh();
                        } catch (err) {
                          toast("error", errorMessage(err));
                        }
                      }}
                    >
                      Delete
                    </button>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </section>

        {system && (
          <section className="stack">
            <h2>Engine room</h2>
            <div className="sys-grid">
              <div className="card" style={{ padding: 12 }}>
                <h3>Compute</h3>
                <div style={{ marginTop: 6 }}>
                  {system.device === "cuda" ? <span className="chip ok">GPU</span> : <span className="chip warn">CPU</span>}{" "}
                  <span className="muted">{system.device_name ?? (system.device === "cpu" ? "CPU (slower separation)" : system.device)}</span>
                </div>
              </div>
              <div className="card" style={{ padding: 12 }}>
                <h3>Separation models</h3>
                <div className="stack" style={{ gap: 4, marginTop: 6 }}>
                  {system.models.map((m) => (
                    <div key={m.name} className="row" title={m.license}>
                      <span className="mono" style={{ fontSize: 12.5 }}>{m.name}</span>
                      <span className="spacer" />
                      <span className={`chip ${m.installed ? "ok" : ""}`}>{m.installed ? "installed" : "not installed"}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="card" style={{ padding: 12 }}>
                <h3>Transcription engines</h3>
                <div className="stack" style={{ gap: 4, marginTop: 6 }}>
                  {system.engines.map((e) => (
                    <div key={e.id} className="muted" style={{ fontSize: 12.5 }} title={e.description}>
                      {e.label}
                    </div>
                  ))}
                </div>
              </div>
              <div className="card" style={{ padding: 12 }}>
                <h3>Formats</h3>
                <div className="muted" style={{ marginTop: 6, fontSize: 12.5 }}>
                  {system.formats.map((f) => f.toUpperCase()).join(", ")}
                  {!system.m4a_supported && <div className="faint">M4A/AAC needs FFmpeg on the server.</div>}
                </div>
              </div>
            </div>
          </section>
        )}
      </main>
    </>
  );
}
