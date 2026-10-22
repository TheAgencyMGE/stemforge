import { describe, expect, it } from "vitest";
import {
  History, cleanup, clampNote, deleteNotes, insertNote, moveNotes, notesInRect, pasteAt, pitchName, quantize,
  resizeNotes, setVelocity, snapTime, snapToScale, transpose, type Grid,
} from "@/lib/notes";
import type { Note } from "@/lib/types";

const n = (id: string, pitch: number, start: number, end: number, extra: Partial<Note> = {}): Note => ({
  id, pitch, start, end, velocity: 80, confidence: 0.8, bends: null, ...extra,
});

const grid: Grid = { bpm: 120, origin: 0.1, beatsPerBar: 4, division: 0.5 }; // eighth notes = 0.25 s, phase 0.1 s

describe("grid snapping", () => {
  it("snaps relative to the detected beat phase", () => {
    expect(snapTime(0.33, grid)).toBeCloseTo(0.35);
    expect(snapTime(0.2, grid)).toBeCloseTo(0.1);
    expect(snapTime(0.34, grid, "floor")).toBeCloseTo(0.1);
    expect(snapTime(0.36, grid, "floor")).toBeCloseTo(0.35);
  });
  it("is a no-op when the grid is off", () => {
    expect(snapTime(0.123, { ...grid, division: 0 })).toBe(0.123);
  });
});

describe("quantize", () => {
  const notes = [n("a", 60, 0.38, 0.6), n("b", 62, 0.9, 1.0)];
  it("moves starts fully and keeps durations", () => {
    const q = quantize(notes, new Set(["a"]), grid, 1);
    expect(q[0].start).toBeCloseTo(0.35);
    expect(q[0].end - q[0].start).toBeCloseTo(0.22);
    expect(q[1]).toBe(notes[1]); // untouched notes keep identity
  });
  it("supports partial strength", () => {
    const q = quantize(notes, new Set(["a"]), grid, 0.5);
    expect(q[0].start).toBeCloseTo(0.365);
  });
  it("can quantize ends too", () => {
    const q = quantize([n("c", 60, 0.11, 0.4)], new Set(["c"]), grid, 1, true);
    expect(q[0].start).toBeCloseTo(0.1);
    expect(q[0].end).toBeCloseTo(0.35);
  });
});

describe("editing operations", () => {
  const notes = [n("a", 60, 1, 2), n("b", 64, 1.5, 2.5), n("c", 67, 3, 4)];
  it("moves a selection together and clamps at the edges", () => {
    const m = moveNotes(notes, new Set(["a", "b"]), -5, 70, 10);
    expect(m[0].start).toBe(0); // clamped by the earliest note
    expect(m[1].start).toBeCloseTo(0.5); // relative timing preserved
    expect(m[1].pitch).toBe(127); // highest note clamped to 127, others shift equally
    expect(m[0].pitch).toBe(123);
  });
  it("resizes without inverting notes", () => {
    const r = resizeNotes(notes, new Set(["a"]), "end", -5, 10);
    expect(r[0].end).toBeGreaterThan(r[0].start);
    const s = resizeNotes(notes, new Set(["a"]), "start", 0.25, 10);
    expect(s[0].start).toBeCloseTo(1.25);
  });
  it("inserts a user note with no fabricated confidence", () => {
    const { notes: out, note } = insertNote(notes, { pitch: 72, start: 0.5, end: 0.75, velocity: 90, confidence: null, bends: null }, 10);
    expect(out).toHaveLength(4);
    expect(out[0].id).toBe(note.id); // sorted by start
    expect(note.confidence).toBeNull();
  });
  it("deletes, transposes, sets velocity", () => {
    expect(deleteNotes(notes, new Set(["b"])).map((x) => x.id)).toEqual(["a", "c"]);
    expect(transpose(notes, new Set(["c"]), -12)[2].pitch).toBe(55);
    expect(setVelocity(notes, new Set(["a"]), 300)[0].velocity).toBe(127);
  });
  it("snaps out-of-key pitches to the scale", () => {
    const out = snapToScale([n("x", 61, 0, 1), n("y", 60, 0, 1), n("z", 66, 0, 1)], new Set(["x", "y", "z"]), 0, "major");
    expect(out.map((x) => x.pitch)).toEqual([60, 60, 65]);
  });
  it("cleans up short and low-confidence notes but keeps user notes", () => {
    const ns = [n("a", 60, 0, 0.02), n("b", 60, 1, 2, { confidence: 0.1 }), n("c", 60, 3, 4, { confidence: null })];
    expect(cleanup(ns, 0.05, 0.3).map((x) => x.id)).toEqual(["c"]);
  });
  it("selects notes inside a rectangle", () => {
    expect([...notesInRect(notes, 0.9, 1.6, 59, 65)].sort()).toEqual(["a", "b"]);
  });
  it("pastes copies at a new time as user-authored notes", () => {
    const { notes: out, ids } = pasteAt(notes, [notes[0], notes[1]], 5, 10);
    expect(out).toHaveLength(5);
    const pasted = out.filter((x) => ids.has(x.id));
    expect(pasted.map((x) => x.start)).toEqual([5, 5.5]);
    expect(pasted.every((x) => x.confidence === null)).toBe(true);
  });
  it("clamps notes into the song", () => {
    const c = clampNote(n("a", 200, -1, 0.001, { velocity: 0 }), 5);
    expect(c.pitch).toBe(127);
    expect(c.start).toBe(0);
    expect(c.end).toBeGreaterThan(c.start);
    expect(c.velocity).toBe(1);
  });
  it("names pitches", () => {
    expect(pitchName(60)).toBe("C4");
    expect(pitchName(69)).toBe("A4");
  });
});

describe("History", () => {
  it("undoes and redoes snapshots", () => {
    const h = new History<number>();
    h.push(1);
    h.push(2);
    expect(h.undo(3)).toBe(2);
    expect(h.undo(2)).toBe(1);
    expect(h.undo(1)).toBeNull();
    expect(h.redo(1)).toBe(2);
    h.push(9);
    expect(h.canRedo).toBe(false);
  });
});
