"use client";

import { useEffect, useMemo, useState } from "react";
import { useWorkspace } from "./Workspace";
import { api, downloadFromApi, errorMessage } from "@/lib/api";
import { cleanup, removeBends } from "@/lib/notes";
import { flushSaves, useEditor } from "@/lib/store";
import type { EngineId, Job, Track } from "@/lib/types";

const ENGINE_LABEL: Record<EngineId, string> = {
  "basic-pitch": "Basic Pitch (polyphonic)",
  pyin: "pYIN (monophonic melody)",
  "drum-nmf": "Drum NMF (kick/snare/hi-hat)",
};

const PARAM_META: Record<string, { label: string; min: number; max: number; step: number; help: string }> = {
  onset_threshold: { label: "Onset sensitivity threshold", min: 0.05, max: 0.95, step: 0.05, help: "Higher = fewer, more certain note starts" },
  frame_threshold: { label: "Sustain / voicing threshold", min: 0.05, max: 0.95, step: 0.05, help: "Higher = notes end sooner, fewer ghost notes" },
  min_note_ms: { label: "Minimum note length (ms)", min: 10, max: 500, step: 5, help: "Shorter notes are discarded" },
};

export function Inspector() {
  const { project } = useWorkspace();
  const selectedTrackId = useEditor((s) => s.selectedTrackId);
  const track = project.tracks.find((t) => t.id === selectedTrackId);
  return (
    <aside className="inspector" aria-label="Inspector">
      {track ? <TrackInspector key={track.id} track={track} /> : <div className="faint">Select a track.</div>}
      <ProjectSection />
    </aside>
  );
}

function TrackInspector({ track }: { track: Track }) {
  const { project, reload, reloadTrackNotes } = useWorkspace();
  const notes = useEditor((s) => s.notes[track.id]);
  const edit = useEditor((s) => s.edit);
  const toast = useEditor((s) => s.toast);
  const confThreshold = useEditor((s) => s.confThreshold);
  const allowed: EngineId[] = track.is_drum ? ["drum-nmf"] : ["basic-pitch", "pyin"];
  const [engine, setEngine] = useState<EngineId>((track.engine as EngineId) ?? allowed[0]);
  const [params, setParams] = useState<Record<string, number | null>>({ ...track.engine_params });
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [name, setName] = useState(track.display_name);
  const [minLenMs, setMinLenMs] = useState(60);
  const editedLocally = useMemo(() => track.edited || (notes?.length ?? 0) !== track.note_count, [track, notes]);

  // Engine defaults when switching engines (bass keeps a low-frequency band either way).
  function chooseEngine(next: EngineId) {
    setEngine(next);
    if (next === track.engine) setParams({ ...track.engine_params });
    else if (next === "pyin") setParams({ onset_threshold: 0.5, frame_threshold: 0.5, min_note_ms: 90, min_freq: track.stem === "bass" ? 30 : 65, max_freq: track.stem === "bass" ? 500 : 1100 });
    else if (next === "basic-pitch") setParams({ onset_threshold: 0.5, frame_threshold: 0.3, min_note_ms: 80, min_freq: track.stem === "bass" ? 30 : null, max_freq: track.stem === "bass" ? 500 : null });
  }

  // Poll an active re-transcription job
  useEffect(() => {
    if (!job || job.status === "succeeded" || job.status === "failed" || job.status === "cancelled") return;
    const t = setInterval(async () => {
      try {
        const j = await api.getJob(job.id);
        setJob(j);
        if (j.status === "succeeded") {
          await reloadTrackNotes(track.id);
          await reload();
          toast("success", `${track.display_name} re-transcribed with ${ENGINE_LABEL[engine]}`);
        } else if (j.status === "failed") {
          toast("error", `Re-transcription failed: ${j.error}`);
        }
      } catch (e) {
        toast("error", errorMessage(e));
      }
    }, 700);
    return () => clearInterval(t);
  }, [job, reload, reloadTrackNotes, toast, track.id, track.display_name, engine]);

  async function retranscribe() {
    if (editedLocally && !confirm("Re-transcribing replaces your edits on this track with fresh model output. Continue? (Undo history is cleared.)")) return;
    setBusy(true);
    try {
      await flushSaves().catch(() => undefined);
      const j = await api.retranscribe(project.id, track.id, engine, params);
      setJob(j);
    } catch (e) {
      toast("error", errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  const stemArtifact = project.artifacts.find((a) => a.kind === "stem" && a.stem === track.stem);
  const summary = track.transcription_summary;
  const jobActive = job && (job.status === "queued" || job.status === "running");
  const lowCount = (notes ?? []).filter((n) => n.confidence !== null && n.confidence < confThreshold).length;

  return (
    <>
      <section>
        <h3>Track</h3>
        <input
          type="text" aria-label="Track name" value={name} maxLength={64}
          onChange={(e) => setName(e.target.value)}
          onBlur={async () => {
            if (!name.trim() || name === track.display_name) return;
            try {
              await api.patchTrack(project.id, track.id, { display_name: name.trim() });
              await reload();
            } catch (e) {
              toast("error", errorMessage(e));
            }
          }}
        />
        <dl className="kv">
          <dt>Separated by</dt><dd>{track.source_model.replace("demucs:", "")}</dd>
          <dt>Level vs mix</dt><dd>{track.stats.relative_db != null ? `${track.stats.relative_db} dB` : "–"}</dd>
          <dt>Peak</dt><dd>{track.stats.peak_db} dBFS</dd>
          {summary?.count != null && <><dt>Notes (model)</dt><dd>{summary.count}</dd></>}
          {summary?.mean_confidence != null && <><dt>Mean confidence</dt><dd>{summary.mean_confidence.toFixed(2)}</dd></>}
          <dt>Below threshold</dt><dd>{lowCount}</dd>
        </dl>
        {track.warnings.map((w) => <div key={w} className="warnbox">{w}</div>)}
      </section>

      <section>
        <h3>Transcription</h3>
        {track.transcription_status === "skipped" ? (
          <div className="faint">{summary?.detail ?? "Skipped."} Nothing was generated for this stem.</div>
        ) : (
          <>
            {track.engine_reason && (
              <div className="reason">
                <strong style={{ color: "var(--text)" }}>Why {ENGINE_LABEL[(track.engine as EngineId) ?? allowed[0]]}?</strong> {track.engine_reason}
              </div>
            )}
            <label className="field">
              Engine
              <select value={engine} onChange={(e) => chooseEngine(e.target.value as EngineId)} disabled={allowed.length < 2}>
                {allowed.map((id) => <option key={id} value={id}>{ENGINE_LABEL[id]}</option>)}
              </select>
            </label>
            {Object.entries(PARAM_META).map(([k, m]) =>
              params[k] === undefined ? null : (
                <label key={k} className="field" title={m.help}>
                  {m.label}
                  <span className="slider-row">
                    <input type="range" min={m.min} max={m.max} step={m.step} value={Number(params[k] ?? m.min)} onChange={(e) => setParams((p) => ({ ...p, [k]: Number(e.target.value) }))} />
                    <output>{Number(params[k]).toFixed(k === "min_note_ms" ? 0 : 2)}</output>
                  </span>
                </label>
              ),
            )}
            {"min_freq" in params && (
              <div className="row">
                <label className="field" style={{ flex: 1 }}>
                  Min Hz
                  <input type="number" min={20} max={8000} value={params.min_freq ?? ""} placeholder="auto" onChange={(e) => setParams((p) => ({ ...p, min_freq: e.target.value ? Number(e.target.value) : null }))} />
                </label>
                <label className="field" style={{ flex: 1 }}>
                  Max Hz
                  <input type="number" min={20} max={8000} value={params.max_freq ?? ""} placeholder="auto" onChange={(e) => setParams((p) => ({ ...p, max_freq: e.target.value ? Number(e.target.value) : null }))} />
                </label>
              </div>
            )}
            <button className="btn primary" disabled={busy || !!jobActive} onClick={retranscribe}>
              {jobActive ? "Re-transcribing…" : "Re-transcribe this track"}
            </button>
            {job && (
              <div className="stack" style={{ gap: 4 }} aria-live="polite">
                <div className="progress"><div style={{ width: `${job.progress * 100}%` }} /></div>
                <div className="faint" style={{ fontSize: 12 }}>{job.status === "failed" ? `Failed: ${job.error}` : job.message ?? job.status}</div>
              </div>
            )}
            <div className="faint" style={{ fontSize: 12 }}>
              Expensive model inference is cached, so changing thresholds re-runs in about a second.
            </div>
          </>
        )}
      </section>

      {track.transcription_status === "done" && (
        <section>
          <h3>Clean up</h3>
          <button className="btn sm" disabled={!lowCount} onClick={() => edit(track.id, (cur) => cleanup(cur, 0, confThreshold))}>
            Remove {lowCount} low-confidence note{lowCount === 1 ? "" : "s"}
          </button>
          <div className="row">
            <button className="btn sm" onClick={() => edit(track.id, (cur) => cleanup(cur, minLenMs / 1000, 0))}>Remove notes shorter than</button>
            <input type="number" min={5} max={1000} value={minLenMs} onChange={(e) => setMinLenMs(Number(e.target.value))} aria-label="Minimum length in ms" style={{ width: 64, height: 26 }} />
            <span className="faint">ms</span>
          </div>
          {!track.is_drum && (
            <button className="btn sm" onClick={() => edit(track.id, (cur) => removeBends(cur, new Set(cur.map((n) => n.id))))}>Remove all pitch bends</button>
          )}
          <button
            className="btn sm"
            disabled={!editedLocally}
            onClick={async () => {
              if (!confirm("Discard all edits on this track and restore the model's notes?")) return;
              try {
                await flushSaves().catch(() => undefined);
                const doc = await api.resetNotes(project.id, track.id);
                useEditor.getState().setTrackNotes(track.id, doc.notes, doc.version);
                toast("info", "Restored the generated notes");
              } catch (e) {
                toast("error", errorMessage(e));
              }
            }}
          >
            Reset to generated notes
          </button>
        </section>
      )}

      <section>
        <h3>Download</h3>
        <div className="row wrap">
          {stemArtifact && (
            <a className="btn sm" href={stemArtifact.url} download={stemArtifact.filename}>Stem (FLAC 24-bit)</a>
          )}
          {track.transcription_status === "done" && (
            <button
              className="btn sm"
              onClick={async () => {
                try {
                  await flushSaves();
                  await downloadFromApi(`/projects/${project.id}/export/midi?stems=${track.stem}`, `${track.stem}.mid`);
                } catch (e) {
                  toast("error", errorMessage(e));
                }
              }}
            >
              MIDI (with edits)
            </button>
          )}
        </div>
      </section>
    </>
  );
}

function ProjectSection() {
  const { project, reload } = useWorkspace();
  const toast = useEditor((s) => s.toast);
  const has6 = project.tracks.some((t) => t.stem === "guitar" || t.stem === "piano");
  return (
    <section>
      <h3>Project</h3>
      {!has6 && (
        <>
          <button
            className="btn sm"
            onClick={async () => {
              if (!confirm("Run htdemucs_6s to add guitar and piano stems? The existing vocals/drums/bass are reused; the 'Other' stem is replaced by the 6-stem residual and re-transcribed.")) return;
              try {
                await flushSaves().catch(() => undefined);
                await api.reprocess(project.id, true);
                await reload();
              } catch (e) {
                toast("error", errorMessage(e));
              }
            }}
          >
            Add guitar + piano stems
          </button>
          <div className="faint" style={{ fontSize: 12 }}>Experimental: upstream reports heavy bleed on the piano stem.</div>
        </>
      )}
      <div className="faint" style={{ fontSize: 12 }}>
        Edits save automatically. Use Export → Full project to keep a portable copy with all stems and edits.
      </div>
    </section>
  );
}
