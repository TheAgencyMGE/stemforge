"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { AppHeader } from "./Brand";
import { Workspace } from "./Workspace";
import { api, ApiError, errorMessage } from "@/lib/api";
import { fmtEta, fmtTime } from "@/lib/format";
import type { Job, ProjectDetail } from "@/lib/types";

const STAGE_LABEL: Record<string, string> = {
  decode: "Decode",
  analyze: "Analyze",
  separate: "Separate",
  render: "Render",
  transcribe: "Transcribe",
  midi: "MIDI",
};
const STAGE_HINT: Record<string, string> = {
  decode: "Validate and decode to 44.1 kHz stereo",
  analyze: "Tempo, beat grid, key and sections",
  separate: "HTDemucs source separation",
  render: "Waveforms, previews and pitch spectrograms",
  transcribe: "Per-stem note transcription",
  midi: "Write and re-parse Standard MIDI Files",
};

export function ProjectView({ id }: { id: string }) {
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [error, setError] = useState<{ status: number; message: string } | null>(null);

  const load = useCallback(async () => {
    try {
      const p = await api.getProject(id);
      setProject(p);
      setError(null);
      return p;
    } catch (e) {
      setError({ status: e instanceof ApiError ? e.status : 0, message: errorMessage(e) });
      return null;
    }
  }, [id]);

  useEffect(() => {
    // Initial fetch from the API (external system); state is set when the response arrives.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, [load]);

  // Poll while the processing job is active.
  const active = project && (project.status === "queued" || project.status === "processing");
  useEffect(() => {
    if (!active) return;
    const t = setInterval(load, 1000);
    return () => clearInterval(t);
  }, [active, load]);

  if (error && !project) {
    return (
      <>
        <AppHeader />
        <div className="dash">
          <div className="errbox" role="alert">
            {error.status === 404 ? "This project doesn't exist (it may have expired or been deleted)." : error.message}
          </div>
          <Link href="/" className="btn" style={{ justifySelf: "start" }}>
            Back to projects
          </Link>
        </div>
      </>
    );
  }
  if (!project) return <div className="loading-screen">Loading project…</div>;
  if (project.status === "ready") return <Workspace initial={project} reload={load} />;
  return <ProcessingDashboard project={project} reload={load} />;
}

function ProcessingDashboard({ project, reload }: { project: ProjectDetail; reload: () => Promise<unknown> }) {
  const job = project.job;
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const failed = project.status === "failed" || project.status === "cancelled";

  async function act(fn: () => Promise<Job>) {
    setBusy(true);
    setErr(null);
    try {
      await fn();
      await reload();
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  const trimmed = project.options.trim_start || project.options.trim_end;
  return (
    <>
      <AppHeader>
        <Link href="/" className="btn ghost">All projects</Link>
      </AppHeader>
      <main className="dash">
        <div className="row">
          <div>
            <h1>{project.name}</h1>
            <div className="faint num" style={{ marginTop: 4 }}>
              {project.original_filename}
              {project.options.source_duration ? ` · ${fmtTime(project.options.source_duration)}` : ""}
              {trimmed ? ` · section ${fmtTime(project.options.trim_start ?? 0)}–${project.options.trim_end ? fmtTime(project.options.trim_end) : "end"}` : ""}
              {project.options.six_stem ? " · 6-stem" : ""}
            </div>
          </div>
          <span className="spacer" />
          {job && !failed && (
            <button className="btn" disabled={busy} onClick={() => act(() => api.cancelJob(job.id))}>
              Cancel
            </button>
          )}
          {job && failed && (
            <button className="btn primary" disabled={busy} onClick={() => act(() => api.retryJob(job.id))}>
              {project.status === "cancelled" ? "Resume" : "Retry"}
            </button>
          )}
        </div>

        {job && (
          <div className="card" style={{ padding: 16 }} aria-live="polite">
            <div className="row" style={{ marginBottom: 10 }}>
              <strong>{job.status === "queued" ? "Waiting for a worker…" : failed ? "Stopped" : job.message ?? "Working…"}</strong>
              <span className="spacer" />
              <span className="num muted">{Math.round(job.progress * 100)}%</span>
            </div>
            <div className={`progress ${job.status === "queued" ? "indeterminate" : ""}`} role="progressbar" aria-valuenow={Math.round(job.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
              <div style={{ width: `${job.progress * 100}%` }} />
            </div>
            {fmtEta(job.eta_seconds) && <div className="faint" style={{ marginTop: 6, fontSize: 12 }}>{fmtEta(job.eta_seconds)}</div>}
            {job.status === "queued" && job.attempts > 0 && job.message && <div className="faint" style={{ marginTop: 6 }}>{job.message}</div>}
            <div className="stage-list" style={{ marginTop: 14 }}>
              {job.stage_order.map((name) => {
                const st = job.stages[name];
                const status = st?.status ?? (failed && job.stage === name ? "failed" : "pending");
                const cls = failed && job.stage === name && status === "running" ? "failed" : status;
                return (
                  <div key={name} className={`stage ${cls}`}>
                    <span className="dot" aria-hidden="true">{status === "done" || status === "skipped" ? "✓" : cls === "failed" ? "!" : ""}</span>
                    <span className="name">{STAGE_LABEL[name] ?? name}</span>
                    <span className="detail">
                      {st?.message ?? STAGE_HINT[name]}
                      {status === "skipped" && " (resumed)"}
                    </span>
                    <span className="num faint" style={{ fontSize: 12 }}>
                      {status === "running" ? `${Math.round((st?.progress ?? 0) * 100)}%` : st?.seconds != null ? `${st.seconds.toFixed(1)} s` : ""}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
        {(project.error || job?.error) && (
          <div className="errbox" role="alert">
            <strong>{project.status === "cancelled" ? "Cancelled." : "Processing failed."}</strong> {project.error ?? job?.error}
            {project.status === "failed" && (
              <div className="faint" style={{ marginTop: 6 }}>
                Finished stages are kept, so Retry resumes where it stopped. Errors about the file itself (format, length) need a
                new upload.
              </div>
            )}
          </div>
        )}
        {err && <div className="errbox" role="alert">{err}</div>}
      </main>
    </>
  );
}
