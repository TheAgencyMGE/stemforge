"use client";

import { useEffect, useMemo, useState } from "react";
import { useWorkspace } from "./Workspace";
import { api, downloadFromApi, errorMessage } from "@/lib/api";
import { cleanup, removeBends, revertOctaves, tablatureEstimate } from "@/lib/notes";
import { flushSaves, useEditor } from "@/lib/store";
import type { EngineId, Job, Note, Track } from "@/lib/types";

const ENGINE_LABEL: Record<EngineId, string> = {
  "basic-pitch": "Basic Pitch (polyphonic)",
  pyin: "pYIN (monophonic melody)",
  "drum-nmf": "Drum NMF (kit classes)",
  "onset-events": "Onset events (one percussion voice)",
  transkun: "Transkun (piano, research engine)",
  yourmt3: "YourMT3+ (multi-instrument, research engine)",
  muscriptor: "MuScriptor (multi-instrument, research engine)",
};

const RESEARCH_NOTE: Partial<Record<EngineId, string>> = {
  transkun: "Runs in its own environment. Code MIT; the checkpoint has no stated licence. No per-note confidence.",
  yourmt3: "Runs in its own environment. Code GPL-3.0, checkpoint Apache-2.0 per its card. Chosen automatically, it transcribes the whole mix and keeps the notes this source plays. No velocity output (taken from the stem's loudness) and no per-note confidence.",
  muscriptor: "Runs in its own environment. Weights CC BY-NC 4.0: non-commercial research only. No velocity output (taken from the stem's loudness) and no per-note confidence.",
};

const ISOLATION_LABEL: Record<string, string> = {
  broad: "Separated directly by the source-separation model",
  targeted: "Extracted from a parent stem by a targeted split (target + complement = parent)",
  residual: "What remained after every supported extraction; it holds one identified instrument",
  group: "Several instruments that no bundled model can separate from each other",
  unresolved: "Audible but not identified with confidence",
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
  const fallback: EngineId[] = track.stem === "percussion" ? ["onset-events"] : track.is_drum ? ["drum-nmf"] : ["basic-pitch", "pyin"];
  const allowed: EngineId[] = track.allowed_engines?.length ? track.allowed_engines : fallback;
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
          <dt>Separated by</dt><dd>{track.source_model.replace(/^demucs:/, "")}</dd>
          <dt>Level vs mix</dt><dd>{track.stats.relative_db != null ? `${track.stats.relative_db} dB` : "–"}</dd>
          <dt>Peak</dt><dd>{track.stats.peak_db} dBFS</dd>
          {summary?.count != null && <><dt>Notes (model)</dt><dd>{summary.count}</dd></>}
          {summary?.mean_confidence != null && <><dt>Mean confidence</dt><dd>{summary.mean_confidence.toFixed(2)}</dd></>}
          <dt>Below threshold</dt><dd>{lowCount}</dd>
        </dl>
        {track.stats.source && <SourceEvidence source={track.stats.source} />}
        {track.stats.render_fidelity && <RenderFidelity track={track} />}
        {track.warnings.map((w) => <div key={w} className="warnbox">{w}</div>)}
        {summary?.suggestions?.map((s) => (
          <div key={s.code} className={s.severity === "info" ? "reason" : "warnbox"}>
            <strong>{s.severity === "info" ? "Note" : "Review"} · {s.code.replaceAll("_", " ")}</strong>
            <div>{s.message}</div>
          </div>
        ))}
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
                {allowed.map((id) => <option key={id} value={id}>{ENGINE_LABEL[id] ?? id}</option>)}
              </select>
              {RESEARCH_NOTE[engine] && <div className="faint" style={{ fontSize: 12 }}>{RESEARCH_NOTE[engine]}</div>}
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
          {(notes ?? []).some((n) => n.raw_pitch != null) && (
            <button className="btn sm" onClick={() => edit(track.id, (cur) => revertOctaves(cur))}>
              Restore pitches moved by the octave check
            </button>
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

      {track.stem === "guitar" && (notes?.length ?? 0) > 0 && <GuitarTab notes={notes ?? []} />}

      <section>
        <h3>Download</h3>
        <div className="row wrap">
          {stemArtifact && (
            <>
              <button
                className="btn sm"
                title="Leading and trailing silence removed. The original-timeline start is in the file's response header and in zip exports' manifest."
                onClick={async () => {
                  try {
                    await downloadFromApi(`/projects/${project.id}/tracks/${track.id}/audio?mode=trimmed`, `${track.stem}-trimmed.flac`);
                  } catch (e) {
                    toast("error", errorMessage(e));
                  }
                }}
              >
                FLAC (trimmed)
              </button>
              <a className="btn sm" href={stemArtifact.url} download={stemArtifact.filename} title="Full length, sample-aligned with every other stem and the original file.">FLAC (full timeline)</a>
            </>
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

const STEP_TEXT: Record<string, (e: Record<string, unknown>) => string | null> = {
  family_check: (e) => e.accepted ? null
    : `${String(e.stem)}: not present. ${e.family_score == null ? "Energy" : `Tag evidence ${Math.round(Number(e.family_score) * 100)}%`}, audible ${Number((e.activity as { active_seconds?: number })?.active_seconds ?? 0).toFixed(1)} s.`,
  targeted: (e) => e.error ? `${String(e.model)}: could not run (${String(e.error)}).`
    : `${String(e.target)} via ${String(e.model)}: ${e.accepted ? "extracted" : "tried, not kept"} (own tag ${Math.round(Number(e.target_tag) * 100)}%).`,
  reassign: (e) => `${String(e.from)} stem held only bleed (${String(e.rel_db)} dB); merged into ${String(e.to)}.`,
  internal_residual: (e) => `${String(e.from)} stem: ${String(e.reason)} (${String(e.rel_db)} dB); kept out of the track list.`,
  residual_stop: (e) => `Stopped: ${String(e.reason)}.`,
};

function DiscoverySummary() {
  const { project } = useWorkspace();
  const d = project.analysis?.discovery;
  if (!d) return null;
  const residual = project.artifacts.find((a) => a.kind === "residual");
  const lines = d.log.map((e) => STEP_TEXT[e.step]?.(e as Record<string, unknown>)).filter((x): x is string => !!x);
  return (
    <details className="card" style={{ padding: "6px 10px" }}>
      <summary style={{ cursor: "pointer", fontSize: 13 }}>
        Source discovery · {project.tracks.filter((t) => t.stats.source).length} sources · mix rebuilt to {d.reconstruction.residual_db} dB
      </summary>
      <div className="faint" style={{ fontSize: 12, marginTop: 6 }}>
        Tagger: {d.tagger === "unavailable" ? "not installed (energy-only decisions)" : d.tagger}. Each line is a decision and the evidence behind it.
      </div>
      <ul style={{ fontSize: 12, margin: "6px 0 0 16px", padding: 0 }}>
        {lines.map((l, i) => <li key={i}>{l}</li>)}
      </ul>
      {residual && (
        <a className="btn sm" style={{ marginTop: 6 }} href={residual.url} download={residual.filename}
          title="Audio that no source could be credited with on the evidence (for example harmonics that leaked into the residual). It is not a track, but nothing is thrown away.">
          Unattributed residual (FLAC, {d.reconstruction.internal_residual_db} dB)
        </a>
      )}
    </details>
  );
}

function RenderFidelity({ track }: { track: Track }) {
  const { project } = useWorkspace();
  const f = track.stats.render_fidelity!;
  const render = project.artifacts.find((a) => a.kind === "midi_render" && a.stem === track.stem);
  const pct = (v: number | null) => (v == null ? "–" : `${Math.round(v * 100)}%`);
  return (
    <div className="reason" style={{ marginBottom: 8 }}>
      <strong style={{ color: "var(--text)" }}>MIDI vs this stem</strong>
      <dl className="kv" style={{ marginTop: 6 }}>
        <dt title="Pitch classes and harmony, largely independent of the instrument sound">Pitch content match</dt><dd>{pct(f.chroma_similarity)}</dd>
        <dt title="Correlation of note attacks over time">Rhythm match</dt><dd>{pct(f.onset_correlation)}</dd>
        <dt title="Overall spectral difference, including timbre, effects and mix">Sound difference</dt><dd>{f.logmel_l1_db == null ? "–" : `${f.logmel_l1_db} dB`}</dd>
      </dl>
      <div className="faint" style={{ fontSize: 12 }}>
        The MIDI was rendered through a General MIDI soundfont and compared with this separated stem, bleed included. It says how closely the MIDI
        reproduces the stem, not how accurate the transcription is: on the BabySlakh benchmark these numbers did not reliably pick the more accurate engine.
        The sound difference also includes the instrument sound, which no MIDI file carries.
      </div>
      {render && <audio controls preload="none" src={render.url} style={{ width: "100%", marginTop: 6 }} aria-label="MIDI rendered through the soundfont" />}
    </div>
  );
}

function SourceEvidence({ source }: { source: NonNullable<Track["stats"]["source"]> }) {
  const a = source.activity;
  return (
    <div className="reason" style={{ marginBottom: 8 }}>
      <strong style={{ color: "var(--text)" }}>
        {source.status === "unresolved" ? "Unresolved source" : source.display}
      </strong>
      <dl className="kv" style={{ marginTop: 6 }}>
        <dt>Identity evidence</dt>
        <dd>{source.confidence == null ? "energy only (tagger not installed)" : `${Math.round(source.confidence * 100)}% tag probability`}</dd>
        <dt>Isolation</dt><dd title={ISOLATION_LABEL[source.isolation]}>{source.isolation}</dd>
        <dt>Audible</dt>
        <dd>{a.active_seconds.toFixed(1)} s{a.first_active != null ? ` (from ${a.first_active.toFixed(1)} s)` : ""}</dd>
        {a.median_active_rel_db != null && <><dt>Level when playing</dt><dd>{a.median_active_rel_db} dB vs mix</dd></>}
        <dt>Audio from</dt><dd>{source.separation}</dd>
      </dl>
      <div className="faint" style={{ fontSize: 12 }}>{ISOLATION_LABEL[source.isolation]}.</div>
      {source.candidates.length > 0 && (
        <div className="faint" style={{ fontSize: 12, marginTop: 4 }}>
          Tag candidates: {source.candidates.slice(0, 4).map((c) => `${c.name} ${Math.round(c.score * 100)}%`).join(", ")}
        </div>
      )}
    </div>
  );
}

function GuitarTab({ notes }: { notes: Note[] }) {
  const tab = tablatureEstimate(notes);
  return (
    <section>
      <h3>Tablature estimate</h3>
      <div className="reason">{tab.disclaimer} Tuning {tab.tuning}. {tab.assigned} assigned, {tab.unassigned} left unassigned where two notes needed the same string.</div>
      <details>
        <summary className="faint" style={{ fontSize: 12, cursor: "pointer" }}>Fret assignments</summary>
        <div className="mono" style={{ fontSize: 12, maxHeight: 160, overflow: "auto" }}>
          {tab.events.slice(0, 80).map((e) => (
            <div key={e.note_id}>{e.string == null ? "— no free string" : `string ${e.string} fret ${e.fret}`}</div>
          ))}
        </div>
      </details>
    </section>
  );
}

function ProjectSection() {
  const { project, reload } = useWorkspace();
  const toast = useEditor((s) => s.toast);
  const has6 = project.tracks.some((t) => t.stem === "guitar" || t.stem === "piano");
  const tried6 = project.options.six_stem || (project.analysis?.discovery?.log ?? []).some((e) => e.step === "targeted" && e.model === "htdemucs_6s");
  return (
    <section>
      <h3>Project</h3>
      <DiscoverySummary />
      {!has6 && !tried6 && (
        <>
          <button
            className="btn sm"
            onClick={async () => {
              if (!confirm("Force a guitar/piano extraction pass with htdemucs_6s? Discovery skipped it because the tags did not indicate guitar or piano. A part is only kept if its own tags confirm it.")) return;
              try {
                await flushSaves().catch(() => undefined);
                await api.reprocess(project.id, { six_stem: true });
                await reload();
              } catch (e) {
                toast("error", errorMessage(e));
              }
            }}
          >
            Try guitar / piano extraction anyway
          </button>
          <div className="faint" style={{ fontSize: 12 }}>Experimental: upstream reports heavy bleed on the piano stem.</div>
        </>
      )}
      {!project.options.drum_kit && project.tracks.some((t) => t.stem === "drums") && (
        <button
          className="btn sm"
          onClick={async () => {
            if (!confirm("Split the drum stem into kick, snare, hat and residual masks? These are for listening. Drum MIDI stays on the Drums track.")) return;
            try {
              await flushSaves().catch(() => undefined);
              await api.reprocess(project.id, { drum_kit: true });
              await reload();
            } catch (e) {
              toast("error", errorMessage(e));
            }
          }}
        >
          Split drum stem into kit pieces
        </button>
      )}
      {!project.options.transcribe_mix && (
        <button
          className="btn sm"
          onClick={async () => {
            if (!confirm("Transcribe the original mix with Basic Pitch, as one extra track? This does not replace the stem MIDI.")) return;
            try {
              await flushSaves().catch(() => undefined);
              await api.reprocess(project.id, { transcribe_mix: true });
              await reload();
            } catch (e) {
              toast("error", errorMessage(e));
            }
          }}
        >
          Also transcribe the full mix
        </button>
      )}
      {!project.options.compare_engines && (
        <button
          className="btn sm"
          onClick={async () => {
            if (!confirm("Run a second engine on bass and vocals and list disagreements? Notes are not averaged or deleted. This is slower.")) return;
            try {
              await flushSaves().catch(() => undefined);
              await api.reprocess(project.id, { compare_engines: true });
              await reload();
            } catch (e) {
              toast("error", errorMessage(e));
            }
          }}
        >
          Compare a second engine on bass and vocals
        </button>
      )}
      {project.analysis?.meter?.time_signature && (
        <div className="faint" style={{ fontSize: 12 }}>
          Beat accents suggest {project.analysis.meter.time_signature} (confidence {project.analysis.meter.confidence.toFixed(2)}).
          The grid uses the time signature in the transport until you change it.
        </div>
      )}
      <div className="faint" style={{ fontSize: 12 }}>
        Edits save automatically. Quantize and octave restores are undoable edits. Reset brings back the last model notes, including any marked octave correction.
      </div>
    </section>
  );
}
