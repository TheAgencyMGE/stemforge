import type { Note } from "./types";

/** Musical grid derived from tempo and the first detected beat (all in seconds). */
export interface Grid {
  bpm: number;
  origin: number; // time of a beat (first detected beat), used as grid phase
  beatsPerBar: number;
  /** Grid division in beats: 1 = quarter, 0.5 = eighth, 1/3 = eighth-triplet, 0 = off */
  division: number;
}

export const DIVISIONS: { label: string; value: number }[] = [
  { label: "Off", value: 0 },
  { label: "1/1 bar", value: 4 },
  { label: "1/2", value: 2 },
  { label: "1/4", value: 1 },
  { label: "1/8", value: 0.5 },
  { label: "1/8T", value: 1 / 3 },
  { label: "1/16", value: 0.25 },
  { label: "1/16T", value: 1 / 6 },
  { label: "1/32", value: 0.125 },
];

export const MIN_NOTE = 0.01;

export function secondsPerBeat(g: Grid): number {
  return 60 / (g.bpm > 0 ? g.bpm : 120);
}

export function gridStep(g: Grid): number {
  return g.division > 0 ? g.division * secondsPerBeat(g) : 0;
}

export function snapTime(t: number, g: Grid, mode: "round" | "floor" = "round"): number {
  const step = gridStep(g);
  if (step <= 0) return t;
  const k = (t - g.origin) / step;
  const n = mode === "floor" ? Math.floor(k + 1e-9) : Math.round(k);
  return Math.max(0, g.origin + n * step);
}

export function clampNote(n: Note, duration: number): Note {
  const start = Math.max(0, Math.min(n.start, duration - MIN_NOTE));
  const end = Math.max(start + MIN_NOTE, Math.min(n.end, duration));
  return {
    ...n,
    start: round4(start),
    end: round4(end),
    pitch: Math.max(0, Math.min(127, Math.round(n.pitch))),
    velocity: Math.max(1, Math.min(127, Math.round(n.velocity))),
  };
}

export function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

export function sortNotes(notes: Note[]): Note[] {
  return [...notes].sort((a, b) => a.start - b.start || a.pitch - b.pitch);
}

let counter = 0;
export function newNoteId(): string {
  counter = (counter + 1) % 1_000_000;
  const rand = Math.floor(Math.random() * 0xffffff).toString(16).padStart(6, "0");
  return `u${Date.now().toString(36)}${counter.toString(36)}${rand}`.slice(0, 32);
}

/** Quantize note starts (and optionally ends) towards the grid. strength 0..1 */
export function quantize(notes: Note[], ids: Set<string>, g: Grid, strength = 1, ends = false): Note[] {
  if (gridStep(g) <= 0) return notes;
  const s = Math.max(0, Math.min(1, strength));
  return notes.map((n) => {
    if (!ids.has(n.id)) return n;
    const len = n.end - n.start;
    const qs = n.start + (snapTime(n.start, g) - n.start) * s;
    let qe = qs + len;
    if (ends) {
      const target = snapTime(qe, g);
      qe = qe + (target - qe) * s;
      if (qe - qs < gridStep(g) * 0.5) qe = qs + Math.max(gridStep(g) * s, MIN_NOTE);
    }
    return { ...n, start: round4(Math.max(0, qs)), end: round4(Math.max(qs + MIN_NOTE, qe)) };
  });
}

export function moveNotes(notes: Note[], ids: Set<string>, dt: number, dp: number, duration: number): Note[] {
  // Clamp the whole selection together so relative timing is preserved at the edges.
  const sel = notes.filter((n) => ids.has(n.id));
  if (!sel.length) return notes;
  const minStart = Math.min(...sel.map((n) => n.start));
  const maxEnd = Math.max(...sel.map((n) => n.end));
  const minP = Math.min(...sel.map((n) => n.pitch));
  const maxP = Math.max(...sel.map((n) => n.pitch));
  const t = Math.max(-minStart, Math.min(dt, duration - maxEnd));
  const p = Math.max(-minP, Math.min(dp, 127 - maxP));
  return notes.map((n) =>
    ids.has(n.id) ? { ...n, start: round4(n.start + t), end: round4(n.end + t), pitch: n.pitch + p } : n,
  );
}

export function resizeNotes(notes: Note[], ids: Set<string>, edge: "start" | "end", dt: number, duration: number): Note[] {
  return notes.map((n) => {
    if (!ids.has(n.id)) return n;
    if (edge === "end") return { ...n, end: round4(Math.min(duration, Math.max(n.start + MIN_NOTE, n.end + dt))) };
    return { ...n, start: round4(Math.max(0, Math.min(n.end - MIN_NOTE, n.start + dt))), bends: null };
  });
}

export function deleteNotes(notes: Note[], ids: Set<string>): Note[] {
  return notes.filter((n) => !ids.has(n.id));
}

export function insertNote(notes: Note[], n: Omit<Note, "id"> & { id?: string }, duration: number): { notes: Note[]; note: Note } {
  const note = clampNote({ ...n, id: n.id ?? newNoteId() } as Note, duration);
  return { notes: sortNotes([...notes, note]), note };
}

export function setVelocity(notes: Note[], ids: Set<string>, velocity: number): Note[] {
  const v = Math.max(1, Math.min(127, Math.round(velocity)));
  return notes.map((n) => (ids.has(n.id) ? { ...n, velocity: v } : n));
}

export function scaleVelocity(notes: Note[], ids: Set<string>, factor: number, offset = 0): Note[] {
  return notes.map((n) =>
    ids.has(n.id) ? { ...n, velocity: Math.max(1, Math.min(127, Math.round(n.velocity * factor + offset))) } : n,
  );
}

export function transpose(notes: Note[], ids: Set<string>, semitones: number): Note[] {
  return moveNotes(notes, ids, 0, semitones, Infinity);
}

export function removeBends(notes: Note[], ids: Set<string>): Note[] {
  return notes.map((n) => (ids.has(n.id) ? { ...n, bends: null } : n));
}

const MAJOR = [0, 2, 4, 5, 7, 9, 11];
const MINOR = [0, 2, 3, 5, 7, 8, 10];

export function scalePitchClasses(tonicPc: number, mode: "major" | "minor"): Set<number> {
  return new Set((mode === "major" ? MAJOR : MINOR).map((d) => (d + tonicPc) % 12));
}

/** Pitch correction: move out-of-key notes to the nearest in-key pitch (ties go down). */
export function snapToScale(notes: Note[], ids: Set<string>, tonicPc: number, mode: "major" | "minor"): Note[] {
  const pcs = scalePitchClasses(tonicPc, mode);
  return notes.map((n) => {
    if (!ids.has(n.id) || pcs.has(((n.pitch % 12) + 12) % 12)) return n;
    for (let d = 1; d <= 6; d++) {
      if (pcs.has((((n.pitch - d) % 12) + 12) % 12) && n.pitch - d >= 0) return { ...n, pitch: n.pitch - d };
      if (pcs.has((n.pitch + d) % 12) && n.pitch + d <= 127) return { ...n, pitch: n.pitch + d };
    }
    return n;
  });
}

/** Remove notes shorter than minLen and/or with confidence below minConf (user-drawn notes are kept). */
export function cleanup(notes: Note[], minLen: number, minConf: number): Note[] {
  return notes.filter((n) => n.end - n.start >= minLen && (n.confidence === null || n.confidence >= minConf));
}

export function notesInRect(notes: Note[], t0: number, t1: number, p0: number, p1: number): Set<string> {
  const [ta, tb] = t0 < t1 ? [t0, t1] : [t1, t0];
  const [pa, pb] = p0 < p1 ? [p0, p1] : [p1, p0];
  return new Set(notes.filter((n) => n.end >= ta && n.start <= tb && n.pitch >= pa && n.pitch <= pb).map((n) => n.id));
}

export function duplicateAt(notes: Note[], ids: Set<string>, at: number, duration: number): { notes: Note[]; ids: Set<string> } {
  const sel = notes.filter((n) => ids.has(n.id));
  if (!sel.length) return { notes, ids: new Set() };
  const base = Math.min(...sel.map((n) => n.start));
  const copies = sel
    .map((n) => clampNote({ ...n, id: newNoteId(), start: n.start - base + at, end: n.end - base + at, confidence: null }, duration))
    .filter((n) => n.start < duration - MIN_NOTE);
  return { notes: sortNotes([...notes, ...copies]), ids: new Set(copies.map((c) => c.id)) };
}

export const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
export function pitchName(p: number): string {
  return `${NOTE_NAMES[((p % 12) + 12) % 12]}${Math.floor(p / 12) - 1}`;
}

export const DRUM_NAMES: Record<number, string> = {
  35: "Kick 2", 36: "Kick", 37: "Side stick", 38: "Snare", 39: "Clap", 40: "Snare 2", 41: "Tom low",
  42: "Hi-hat closed", 43: "Tom low 2", 44: "Hi-hat pedal", 45: "Tom mid", 46: "Hi-hat open", 47: "Tom mid 2",
  48: "Tom high", 49: "Crash", 50: "Tom high 2", 51: "Ride",
};

/** Undo/redo history of whole-track snapshots (note arrays are immutable, so snapshots share structure). */
export class History<T> {
  private past: T[] = [];
  private future: T[] = [];
  constructor(private limit = 200) {}
  push(state: T) {
    this.past.push(state);
    if (this.past.length > this.limit) this.past.shift();
    this.future = [];
  }
  undo(current: T): T | null {
    const prev = this.past.pop();
    if (prev === undefined) return null;
    this.future.push(current);
    return prev;
  }
  redo(current: T): T | null {
    const next = this.future.pop();
    if (next === undefined) return null;
    this.past.push(current);
    return next;
  }
  get canUndo() {
    return this.past.length > 0;
  }
  get canRedo() {
    return this.future.length > 0;
  }
  clear() {
    this.past = [];
    this.future = [];
  }
}

/** Paste copies of `clip` so the earliest one starts at `at`. Pasted notes are user-authored (no confidence). */
export function pasteAt(notes: Note[], clip: Note[], at: number, duration: number): { notes: Note[]; ids: Set<string> } {
  if (!clip.length) return { notes, ids: new Set() };
  const base = Math.min(...clip.map((n) => n.start));
  const copies = clip
    .map((n) => clampNote({ ...n, id: newNoteId(), start: n.start - base + at, end: n.end - base + at, confidence: null }, duration))
    .filter((n) => n.start < duration - MIN_NOTE);
  return { notes: sortNotes([...notes, ...copies]), ids: new Set(copies.map((c) => c.id)) };
}
