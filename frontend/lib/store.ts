"use client";

import { create } from "zustand";
import { api, ApiError, errorMessage } from "./api";
import { History, type Grid } from "./notes";
import type { Note, ProjectDetail } from "./types";
import type { LaneMix } from "./audioEngine";

export type Tool = "select" | "draw";
export type SaveState = "saved" | "pending" | "saving" | "error" | "conflict";

export const ORIGINAL_LANE = "original";

export interface Toast {
  id: number;
  kind: "info" | "error" | "success";
  text: string;
}

interface EditorState {
  project: ProjectDetail | null;
  notes: Record<string, Note[]>;
  versions: Record<string, number>;
  dirty: Record<string, boolean>;
  saveState: SaveState;
  saveError: string | null;
  selectedTrackId: string | null;
  selection: Set<string>;
  tool: Tool;
  division: number;
  pxPerSec: number;
  scrollX: number; // seconds at the left edge of the timeline
  mix: Record<string, LaneMix>;
  confThreshold: number;
  showConfidence: boolean;
  showSpectrogram: boolean;
  loop: { start: number; end: number } | null;
  compare: "stems" | "original";
  historyTick: number;
  toasts: Toast[];

  setProject: (p: ProjectDetail) => void;
  setTrackNotes: (trackId: string, notes: Note[], version: number) => void;
  edit: (trackId: string, fn: (notes: Note[]) => Note[], opts?: { coalesce?: string }) => void;
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;
  selectTrack: (id: string | null) => void;
  setSelection: (ids: Set<string>) => void;
  set: (partial: Partial<EditorState>) => void;
  setMix: (laneId: string, mix: Partial<LaneMix>) => void;
  toast: (kind: Toast["kind"], text: string) => void;
  dismissToast: (id: number) => void;
  grid: () => Grid;
}

const histories = new Map<string, History<Note[]>>();
let lastCoalesce: { key: string; at: number } | null = null;
let toastId = 0;

function historyFor(trackId: string): History<Note[]> {
  let h = histories.get(trackId);
  if (!h) {
    h = new History<Note[]>(300);
    histories.set(trackId, h);
  }
  return h;
}

export const defaultMix = (laneId: string): LaneMix => ({
  volume: laneId === ORIGINAL_LANE ? 0.9 : 0.9,
  muted: laneId === ORIGINAL_LANE, // stems play by default; the original is for A/B comparison
  soloed: false,
  audio: true,
  midi: false,
});

export const useEditor = create<EditorState>((set, get) => ({
  project: null,
  notes: {},
  versions: {},
  dirty: {},
  saveState: "saved",
  saveError: null,
  selectedTrackId: null,
  selection: new Set(),
  tool: "select",
  division: 0.25,
  pxPerSec: 80,
  scrollX: 0,
  mix: {},
  confThreshold: 0.4,
  showConfidence: true,
  showSpectrogram: true,
  loop: null,
  compare: "stems",
  historyTick: 0,
  toasts: [],

  setProject: (p) => {
    const prev = get().project;
    const mix = { ...get().mix };
    if (!mix[ORIGINAL_LANE]) mix[ORIGINAL_LANE] = defaultMix(ORIGINAL_LANE);
    for (const t of p.tracks) if (!mix[t.id]) mix[t.id] = defaultMix(t.id);
    const first = p.tracks.find((t) => t.transcription_status === "done") ?? p.tracks[0];
    set({
      project: p,
      mix,
      selectedTrackId: prev?.id === p.id ? get().selectedTrackId ?? first?.id ?? null : first?.id ?? null,
    });
  },

  setTrackNotes: (trackId, notes, version) => {
    historyFor(trackId).clear();
    set((s) => ({
      notes: { ...s.notes, [trackId]: notes },
      versions: { ...s.versions, [trackId]: version },
      dirty: { ...s.dirty, [trackId]: false },
      historyTick: s.historyTick + 1,
    }));
  },

  edit: (trackId, fn, opts) => {
    const current = get().notes[trackId] ?? [];
    const next = fn(current);
    if (next === current) return;
    const now = performance.now();
    // Coalesce rapid edits of the same kind (e.g. a drag) into one undo step.
    const coalesce = opts?.coalesce && lastCoalesce?.key === `${trackId}:${opts.coalesce}` && now - lastCoalesce.at < 600;
    if (!coalesce) historyFor(trackId).push(current);
    lastCoalesce = opts?.coalesce ? { key: `${trackId}:${opts.coalesce}`, at: now } : null;
    set((s) => ({
      notes: { ...s.notes, [trackId]: next },
      dirty: { ...s.dirty, [trackId]: true },
      saveState: "pending",
      historyTick: s.historyTick + 1,
    }));
  },

  undo: () => {
    const id = get().selectedTrackId;
    if (!id) return;
    const prev = historyFor(id).undo(get().notes[id] ?? []);
    if (!prev) return;
    lastCoalesce = null;
    set((s) => ({ notes: { ...s.notes, [id]: prev }, dirty: { ...s.dirty, [id]: true }, saveState: "pending", historyTick: s.historyTick + 1 }));
  },

  redo: () => {
    const id = get().selectedTrackId;
    if (!id) return;
    const next = historyFor(id).redo(get().notes[id] ?? []);
    if (!next) return;
    set((s) => ({ notes: { ...s.notes, [id]: next }, dirty: { ...s.dirty, [id]: true }, saveState: "pending", historyTick: s.historyTick + 1 }));
  },

  canUndo: () => {
    const id = get().selectedTrackId;
    return !!id && historyFor(id).canUndo;
  },
  canRedo: () => {
    const id = get().selectedTrackId;
    return !!id && historyFor(id).canRedo;
  },

  selectTrack: (id) => set({ selectedTrackId: id, selection: new Set() }),
  setSelection: (ids) => set({ selection: ids }),
  set: (partial) => set(partial),
  setMix: (laneId, mix) => set((s) => ({ mix: { ...s.mix, [laneId]: { ...(s.mix[laneId] ?? defaultMix(laneId)), ...mix } } })),
  toast: (kind, text) => {
    const id = ++toastId;
    set((s) => ({ toasts: [...s.toasts.slice(-3), { id, kind, text }] }));
    setTimeout(() => get().dismissToast(id), kind === "error" ? 9000 : 4000);
  },
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
  grid: () => {
    const p = get().project;
    const bpm = p?.tempo_bpm ?? p?.analysis?.tempo ?? 120;
    const [num, den] = (p?.time_signature ?? "4/4").split("/").map(Number);
    return {
      bpm,
      origin: p?.analysis?.first_beat ?? 0,
      beatsPerBar: num * (4 / den),
      division: get().division,
    };
  },
}));

let saving: Promise<void> | null = null;

/** Persist every dirty track. Safe to call repeatedly; concurrent calls share one save pass. */
export async function flushSaves(): Promise<void> {
  if (saving) {
    await saving;
  }
  const run = async () => {
    const s = useEditor.getState();
    const p = s.project;
    if (!p) return;
    const ids = Object.keys(s.dirty).filter((k) => s.dirty[k]);
    if (!ids.length) return;
    useEditor.setState({ saveState: "saving" });
    for (const id of ids) {
      const notes = useEditor.getState().notes[id] ?? [];
      const version = useEditor.getState().versions[id] ?? 0;
      try {
        const doc = await api.putNotes(p.id, id, version, notes);
        const still = useEditor.getState().notes[id] === notes;
        useEditor.setState((st) => ({
          versions: { ...st.versions, [id]: doc.version },
          dirty: { ...st.dirty, [id]: !still },
        }));
      } catch (e) {
        if (e instanceof ApiError && e.status === 409) {
          useEditor.setState({ saveState: "conflict", saveError: e.message });
        } else {
          useEditor.setState({ saveState: "error", saveError: errorMessage(e) });
        }
        throw e;
      }
    }
    const anyDirty = Object.values(useEditor.getState().dirty).some(Boolean);
    useEditor.setState({ saveState: anyDirty ? "pending" : "saved", saveError: null });
  };
  saving = run().finally(() => {
    saving = null;
  });
  return saving;
}
