import { describe, expect, it } from "vitest";
import { Midi } from "@tonejs/midi";
import fs from "node:fs";
import path from "node:path";

// Written by backend/tests/golden_midi.py with the same writer as the Soundtrap package. This checks it
// with a browser MIDI parser (@tonejs/midi). It is not Soundtrap's own importer, which is not public.
const FILE = path.resolve(__dirname, "../fixtures/soundtrap-golden.mid");
const BPM = 96;
const OFFSET = 2.5;

describe("Soundtrap-style MIDI through a browser parser", () => {
  const midi = new Midi(fs.readFileSync(FILE));

  it("has one integer tempo and a 4/4 meter", () => {
    expect(midi.header.tempos).toHaveLength(1);
    expect(midi.header.tempos[0].bpm).toBeCloseTo(BPM, 6);
    expect(midi.header.timeSignatures[0].timeSignature).toEqual([4, 4]);
  });

  it("keeps track names, programs and the drum channel", () => {
    const named = midi.tracks.filter((t) => t.notes.length);
    expect(named.map((t) => t.name)).toEqual(["Acoustic piano", "Drum machine"]);
    const [keys, drums] = named;
    expect(keys.instrument.number).toBe(0);
    expect(keys.channel).not.toBe(9);
    expect(drums.channel).toBe(9);
    expect(drums.instrument.percussion).toBe(true);
    expect(drums.notes.map((n) => n.midi).sort()).toEqual([36, 38, 42]);
  });

  it("places notes on the original timeline to within one tick, with velocities", () => {
    const tick = 60 / BPM / midi.header.ppq;
    const keys = midi.tracks.find((t) => t.name === "Acoustic piano")!;
    const want = [[60, 0.0, 0.48, 100], [64, 0.5, 0.98, 64], [67, 1.0123, 1.9, 30]];
    expect(keys.notes).toHaveLength(3);
    keys.notes.forEach((n, i) => {
      const [pitch, start, end, vel] = want[i];
      expect(n.midi).toBe(pitch);
      expect(Math.abs(n.time - (start + OFFSET))).toBeLessThanOrEqual(tick);
      expect(Math.abs(n.time + n.duration - (end + OFFSET))).toBeLessThanOrEqual(tick);
      expect(Math.round(n.velocity * 127)).toBe(vel);
    });
  });

  it("carries the pitch bend of the monophonic phrase", () => {
    const keys = midi.tracks.find((t) => t.name === "Acoustic piano")!;
    const bends = keys.pitchBends.map((b) => b.value);
    expect(Math.max(...bends)).toBeGreaterThan(0.45); // +1 semitone of a ±2 range = 0.5
    expect(bends[bends.length - 1]).toBe(0); // reset after the note
  });
});
