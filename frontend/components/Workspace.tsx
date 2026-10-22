"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { AudioEngine } from "@/lib/audioEngine";
import { api, errorMessage } from "@/lib/api";
import {
  deleteNotes, duplicateAt, gridStep, moveNotes, pasteAt, quantize, transpose, MIN_NOTE,
} from "@/lib/notes";
import { flushSaves, ORIGINAL_LANE, useEditor } from "@/lib/store";
import type { Note, Peaks, ProjectDetail } from "@/lib/types";
import { Transport } from "./Transport";
import { TrackLanes } from "./TrackLanes";
import { PianoRoll } from "./PianoRoll";
import { Inspector } from "./Inspector";
import { ExportDialog } from "./ExportDialog";
import { ShortcutsDialog } from "./ShortcutsDialog";

export interface SpectrogramInfo {
  img: HTMLImageElement;
  hop: number;
  midiLow: number;
  midiHigh: number;
}

export interface WorkspaceCtx {
  engine: AudioEngine;
  project: ProjectDetail;
  duration: number;
  peaks: Record<string, Peaks>; // laneId -> peaks
  spectrograms: Record<string, SpectrogramInfo>;
  /** Lanes with a preview artifact and no entry here are still loading. */
  laneLoad: Record<string, "ready" | "error">;
  reload: () => Promise<unknown>;
  reloadTrackNotes: (trackId: string) => Promise<void>;
  openExport: () => void;
  openShortcuts: () => void;
  clipboard: { current: Note[] };
}

const Ctx = createContext<WorkspaceCtx | null>(null);
export const useWorkspace = () => {
  const c = useContext(Ctx);
  if (!c) throw new Error("useWorkspace outside Workspace");
  return c;
};

function isTyping(e: KeyboardEvent) {
  const t = e.target as HTMLElement | null;
  if (!t) return false;
  const tag = t.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t.isContentEditable;
}

export function Workspace({ initial, reload }: { initial: ProjectDetail; reload: () => Promise<ProjectDetail | null> }) {
  const [engine] = useState(() => new AudioEngine());
  const [clipboard] = useState<{ current: Note[] }>(() => ({ current: [] }));

  const setProject = useEditor((s) => s.setProject);
  const project = useEditor((s) => s.project) ?? initial;
  const [peaks, setPeaks] = useState<Record<string, Peaks>>({});
  const [spectrograms, setSpectrograms] = useState<Record<string, SpectrogramInfo>>({});
  const [laneLoad, setLaneLoad] = useState<Record<string, "ready" | "error">>({});
  const [exportOpen, setExportOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const duration = initial.analysis?.duration ?? 0;

  useEffect(() => {
    setProject(initial);
  }, [initial, setProject]);

  const stemToLane = useMemo(() => {
    const m: Record<string, string> = { original: ORIGINAL_LANE };
    for (const t of initial.tracks) m[t.stem] = t.id;
    return m;
  }, [initial.tracks]);

  // ---- load audio, peaks, spectrograms (once per project) ----
  useEffect(() => {
    const abort = new AbortController();
    const mix = useEditor.getState().mix;
    engine.configure(duration, (laneId) => useEditor.getState().notes[laneId] ?? null);
    for (const a of initial.artifacts) {
      const lane = a.stem ? stemToLane[a.stem] : undefined;
      if (!lane) continue;
      if (a.kind === "preview") {
        const track = initial.tracks.find((t) => t.id === lane);
        engine.addLane(lane, {
          isDrum: track?.is_drum,
          isOriginal: lane === ORIGINAL_LANE,
          mix: mix[lane] ?? { volume: 0.9, muted: lane === ORIGINAL_LANE, soloed: false, audio: true, midi: false },
        });
        engine
          .loadLane(lane, a.url, abort.signal)
          .then(() => setLaneLoad((s) => ({ ...s, [lane]: "ready" })))
          .catch((e) => {
            if (abort.signal.aborted) return;
            setLaneLoad((s) => ({ ...s, [lane]: "error" }));
            useEditor.getState().toast("error", `Couldn't load ${a.stem} audio: ${errorMessage(e)}`);
          });
      } else if (a.kind === "peaks") {
        fetch(a.url, { signal: abort.signal })
          .then((r) => r.json())
          .then((p: Peaks) => setPeaks((s) => ({ ...s, [lane]: p })))
          .catch(() => undefined);
      } else if (a.kind === "spectrogram") {
        const img = new Image();
        img.onload = () =>
          setSpectrograms((s) => ({
            ...s,
            [lane]: {
              img,
              hop: Number(a.meta.hop_seconds ?? 256 / 22050),
              midiLow: Number(a.meta.midi_low ?? 21),
              midiHigh: Number(a.meta.midi_high ?? 108),
            },
          }));
        img.src = a.url;
      }
    }
    return () => abort.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initial.id]);

  useEffect(() => () => engine.dispose(), [engine]);

  const reloadTrackNotes = useCallback(
    async (trackId: string) => {
      const doc = await api.getNotes(initial.id, trackId);
      useEditor.getState().setTrackNotes(trackId, doc.notes, doc.version);
    },
    [initial.id],
  );

  useEffect(() => {
    for (const t of initial.tracks) {
      if (t.transcription_status === "done" || t.note_count > 0) {
        reloadTrackNotes(t.id).catch((e) => useEditor.getState().toast("error", `Couldn't load notes: ${errorMessage(e)}`));
      } else {
        useEditor.getState().setTrackNotes(t.id, [], t.notes_version);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initial.id]);

  // ---- push mixer state to the engine ----
  const mix = useEditor((s) => s.mix);
  const compare = useEditor((s) => s.compare);
  const loop = useEditor((s) => s.loop);
  useEffect(() => {
    for (const [id, m] of Object.entries(mix)) engine.setMix(id, m);
  }, [mix, engine]);
  useEffect(() => engine.setCompare(compare), [compare, engine]);
  useEffect(() => engine.setLoop(loop), [loop, engine]);

  // ---- autosave ----
  const dirty = useEditor((s) => s.dirty);
  const notesState = useEditor((s) => s.notes);
  useEffect(() => {
    if (!Object.values(dirty).some(Boolean)) return;
    const t = setTimeout(() => {
      flushSaves().catch((e) => useEditor.getState().toast("error", `Autosave failed: ${errorMessage(e)}`));
    }, 800);
    return () => clearTimeout(t);
  }, [dirty, notesState]);

  useEffect(() => {
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if (Object.values(useEditor.getState().dirty).some(Boolean)) {
        e.preventDefault();
        void flushSaves();
      }
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, []);

  // ---- keyboard shortcuts ----
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e)) return;
      const st = useEditor.getState();
      const tid = st.selectedTrackId;
      const notes = tid ? st.notes[tid] ?? [] : [];
      const sel = st.selection;
      const mod = e.ctrlKey || e.metaKey;
      const g = st.grid();
      const step = gridStep(g) || 0.01;
      const key = e.key;
      const handled = () => e.preventDefault();

      if (key === " ") {
        handled();
        if (engine.isPlaying) engine.pause();
        else void engine.play();
        return;
      }
      if (key === "Enter" || key === "Home") {
        handled();
        engine.stop();
        st.set({ scrollX: 0 });
        return;
      }
      if (mod && key.toLowerCase() === "z") {
        handled();
        if (e.shiftKey) st.redo();
        else st.undo();
        return;
      }
      if (mod && key.toLowerCase() === "y") {
        handled();
        st.redo();
        return;
      }
      if (mod && key.toLowerCase() === "s") {
        handled();
        flushSaves()
          .then(() => st.toast("success", "All edits saved"))
          .catch((err) => st.toast("error", errorMessage(err)));
        return;
      }
      if (key === "?") {
        handled();
        setShortcutsOpen(true);
        return;
      }
      if (key.toLowerCase() === "c" && !mod) {
        handled();
        st.set({ compare: st.compare === "original" ? "stems" : "original" });
        return;
      }
      if (key === "+" || key === "=") {
        handled();
        st.set({ pxPerSec: Math.min(1200, st.pxPerSec * 1.25) });
        return;
      }
      if (key === "-") {
        handled();
        st.set({ pxPerSec: Math.max(8, st.pxPerSec / 1.25) });
        return;
      }
      if (/^[1-9]$/.test(key) && !mod) {
        const t = project.tracks[Number(key) - 1];
        if (t) {
          handled();
          st.selectTrack(t.id);
        }
        return;
      }
      if (!tid) return;
      if (key.toLowerCase() === "m" && !mod) {
        handled();
        st.setMix(tid, { muted: !st.mix[tid]?.muted });
        return;
      }
      if (key.toLowerCase() === "s" && !mod) {
        handled();
        st.setMix(tid, { soloed: !st.mix[tid]?.soloed });
        return;
      }
      if (key.toLowerCase() === "v" && !mod) return st.set({ tool: "select" });
      if (key.toLowerCase() === "d" && !mod) return st.set({ tool: "draw" });
      if (key === "Escape") {
        st.setSelection(new Set());
        return;
      }
      if (mod && key.toLowerCase() === "a") {
        handled();
        st.setSelection(new Set(notes.map((n) => n.id)));
        return;
      }
      if (mod && key.toLowerCase() === "c") {
        handled();
        clipboard.current = notes.filter((n) => sel.has(n.id));
        if (clipboard.current.length) st.toast("info", `Copied ${clipboard.current.length} note(s)`);
        return;
      }
      if (mod && key.toLowerCase() === "v") {
        handled();
        if (!clipboard.current.length) return;
        let newIds = new Set<string>();
        st.edit(tid, (cur) => {
          const r = pasteAt(cur, clipboard.current, engine.currentTime, duration);
          newIds = r.ids;
          return r.notes;
        });
        st.setSelection(newIds);
        return;
      }
      if (!sel.size && key.toLowerCase() !== "q") return;
      if (key === "Delete" || key === "Backspace") {
        handled();
        st.edit(tid, (cur) => deleteNotes(cur, sel));
        st.setSelection(new Set());
        return;
      }
      if (key.toLowerCase() === "q" && !mod) {
        handled();
        if (gridStep(g) <= 0) return st.toast("info", "Choose a grid division to quantize");
        const ids = sel.size ? sel : new Set(notes.map((n) => n.id));
        st.edit(tid, (cur) => quantize(cur, ids, g, 1, e.shiftKey));
        st.toast("info", `Quantized ${ids.size} note(s) to the grid${e.shiftKey ? " (starts and ends)" : ""}`);
        return;
      }
      if (key === "ArrowUp" || key === "ArrowDown") {
        handled();
        const d = (key === "ArrowUp" ? 1 : -1) * (e.shiftKey ? 12 : 1);
        st.edit(tid, (cur) => transpose(cur, sel, d), { coalesce: "transpose" });
        return;
      }
      if (key === "ArrowLeft" || key === "ArrowRight") {
        handled();
        const d = (key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? step * 4 : step);
        st.edit(tid, (cur) => moveNotes(cur, sel, d, 0, duration), { coalesce: "nudge" });
        return;
      }
      if (mod && key.toLowerCase() === "d") {
        handled();
        const selNotes = notes.filter((n) => sel.has(n.id));
        const end = Math.max(...selNotes.map((n) => n.end));
        const start = Math.min(...selNotes.map((n) => n.start));
        let ids = new Set<string>();
        st.edit(tid, (cur) => {
          const r = duplicateAt(cur, sel, Math.max(end, start + MIN_NOTE), duration);
          ids = r.ids;
          return r.notes;
        });
        st.setSelection(ids);
        return;
      }
      if (key.toLowerCase() === "l" && !mod) {
        handled();
        const selNotes = notes.filter((n) => sel.has(n.id));
        st.set({ loop: { start: Math.min(...selNotes.map((n) => n.start)), end: Math.max(...selNotes.map((n) => n.end)) } });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [engine, duration, project.tracks, clipboard]);

  const ctx: WorkspaceCtx = {
    engine,
    project,
    duration,
    peaks,
    spectrograms,
    laneLoad,
    reload,
    reloadTrackNotes,
    openExport: () => setExportOpen(true),
    openShortcuts: () => setShortcutsOpen(true),
    clipboard,
  };

  const [split, setSplit] = useState(38);
  const centerRef = useRef<HTMLDivElement>(null);

  return (
    <Ctx.Provider value={ctx}>
      <div className="ws">
        <Transport />
        <InfoBar />
        <div className="ws-main">
          <div
            className="ws-center"
            ref={centerRef}
            style={{ ["--split" as string]: `${split}%` }}
          >
            <TrackLanes />
            <div
              className="splitter"
              role="separator"
              aria-orientation="horizontal"
              aria-label="Resize timeline and piano roll"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "ArrowUp") setSplit((s) => Math.max(15, s - 3));
                if (e.key === "ArrowDown") setSplit((s) => Math.min(75, s + 3));
              }}
              onPointerDown={(e) => {
                (e.target as HTMLElement).setPointerCapture(e.pointerId);
                const r = centerRef.current!.getBoundingClientRect();
                const move = (ev: PointerEvent) => setSplit(Math.max(15, Math.min(75, ((ev.clientY - r.top) / r.height) * 100)));
                const up = () => {
                  window.removeEventListener("pointermove", move);
                  window.removeEventListener("pointerup", up);
                };
                window.addEventListener("pointermove", move);
                window.addEventListener("pointerup", up);
              }}
            />
            <PianoRoll />
          </div>
          <Inspector />
        </div>
      </div>
      {exportOpen && <ExportDialog onClose={() => setExportOpen(false)} />}
      {shortcutsOpen && <ShortcutsDialog onClose={() => setShortcutsOpen(false)} />}
    </Ctx.Provider>
  );
}

function InfoBar() {
  const { project, laneLoad } = useWorkspace();
  const a = project.analysis;
  const loading = project.artifacts.filter((x) => x.kind === "preview" && x.stem && !laneLoad[x.stem === "original" ? ORIGINAL_LANE : project.tracks.find((t) => t.stem === x.stem)?.id ?? ""]).length;
  const clipped = (a?.clipped_ratio ?? 0) > 0.001;
  const resid = project.tracks[0]?.stats.mix_residual_db;
  return (
    <div className="infobar" aria-label="Analysis summary">
      <span className="chip" title={`Tempo confidence ${a?.tempo_confidence ?? "–"}`}>
        Detected tempo <span className="num">{a?.tempo ? a.tempo.toFixed(1) : "–"}</span>
      </span>
      {a?.key && (
        <span className="chip" title={`Krumhansl–Schmuckler correlation ${a.key.correlation}; runner-up ${a.key.runner_up} (margin ${a.key.margin})`}>
          Key {a.key.label}
          {a.key.margin < 0.25 && <span className="faint"> / {a.key.runner_up}?</span>}
        </span>
      )}
      {a?.sections?.length ? <span className="chip">{a.sections.length} sections</span> : null}
      {resid != null && (
        <span className="chip" title="Energy of (mix − sum of stems) relative to the mix. Lower is better.">
          Stem residual <span className="num">{resid} dB</span>
        </span>
      )}
      {clipped && <span className="chip warn">Source is clipped ({((a?.clipped_ratio ?? 0) * 100).toFixed(2)}% samples)</span>}
      {project.offset_seconds > 0 && <span className="chip accent">Section starts at {project.offset_seconds.toFixed(2)} s in the original</span>}
      {loading > 0 && <span className="chip accent">Loading {loading} audio lane{loading > 1 ? "s" : ""}…</span>}
      <span className="spacer" />
      {project.expires_at && <span className="faint">Auto-deletes {new Date(project.expires_at.endsWith("Z") ? project.expires_at : project.expires_at + "Z").toLocaleDateString()}</span>}
    </div>
  );
}
