"""Note refinement that needs audio evidence, and MIDI timing fidelity through an independent parser."""

from __future__ import annotations

import io

import numpy as np
import pretty_midi

from stemforge.pipeline import midi, refine

SR = 44100


def _note(nid, pitch, start, end, conf=0.6):
    return {"id": nid, "pitch": pitch, "start": start, "end": end, "velocity": 80, "confidence": conf, "bends": None}


def _sustain(pitch: int, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    f = 440 * 2 ** ((pitch - 69) / 12)
    x = 0.3 * np.sin(2 * np.pi * f * t) * np.exp(-t * 0.3)
    return np.stack([x, x]).astype(np.float32)


def test_a_fragmented_sustain_is_joined():
    audio = _sustain(60, 2.0)
    notes = [_note("a", 60, 0.0, 0.98), _note("b", 60, 1.0, 1.9)]  # model split one held note
    out, merged = refine.merge_split_sustains(notes, audio, SR)
    assert len(out) == 1 and merged == ["a"]
    assert out[0]["end"] == 1.9 and "merged_sustain" in out[0]["flags"]


def test_a_restruck_repeated_note_is_never_joined():
    one = _sustain(60, 0.5)[:, : int(0.48 * SR)]
    audio = np.zeros((2, SR), dtype=np.float32)
    audio[:, : one.shape[1]] = one
    audio[:, int(0.5 * SR): int(0.5 * SR) + one.shape[1]] = one  # new attack 20 ms after the first ends
    notes = [_note("a", 60, 0.0, 0.48), _note("b", 60, 0.5, 0.98)]
    out, merged = refine.merge_split_sustains(notes, audio, SR)
    assert len(out) == 2 and merged == []


def test_cross_source_bleed_goes_to_the_louder_source_only():
    tracks = {
        "piano": [_note("p1", 64, 1.0, 1.5, conf=0.8), _note("p2", 67, 2.0, 2.4, conf=0.5)],
        "other": [_note("o1", 64, 1.01, 1.5, conf=0.3), _note("o2", 67, 2.0, 2.4, conf=0.45),
                  {**_note("u", 64, 1.0, 1.2), "confidence": None}],  # user-drawn: never touched
    }
    dupes = refine.cross_source_duplicates(tracks)
    assert dupes == {"other": ["o1"]}  # o2 is not clearly quieter, so both copies stay


def test_midi_keeps_unquantized_timing_and_offset_through_an_independent_parser(tmp_path):
    notes = [_note("a", 60, 0.1234, 0.5), _note("b", 64, 1.0007, 1.333), _note("c", 67, 2.71828, 3.0)]
    spec = [midi.MidiTrackSpec("Keys", notes, program=0)]
    bpm, offset = 97.3, 12.5  # a trimmed region that starts 12.5 s into the original file
    mf = midi.build_midi(spec, bpm, "4/4", offset)
    path = tmp_path / "t.mid"
    midi.save_and_validate(mf, path, spec, bpm, offset)
    pm = pretty_midi.PrettyMIDI(io.BytesIO(path.read_bytes()))
    got = sorted((n.start, n.end, n.pitch) for n in pm.instruments[0].notes)
    tick = 60.0 / bpm / midi.PPQ
    for (s, e, p), n in zip(got, sorted(notes, key=lambda n: n["start"]), strict=True):
        assert p == n["pitch"]
        assert abs(s - (n["start"] + offset)) <= tick and abs(e - (n["end"] + offset)) <= tick


def test_soundtrap_golden_midi_is_what_the_writer_produces():
    """frontend/tests/unit/midi-parsers.test.ts parses this file with @tonejs/midi (a browser MIDI
    parser). If the writer changes, regenerate it: python -m tests.golden_midi."""
    from tests import golden_midi

    assert golden_midi.GOLDEN.read_bytes() == golden_midi.build()
    # And it passes the backend's own independent re-parse (symusic) and structure checks.
    midi.validate(golden_midi.GOLDEN, golden_midi.specs(), golden_midi.BPM, golden_midi.OFFSET)


def test_tempo_map_puts_a_changing_tempo_on_the_grid_and_keeps_real_time(tmp_path):
    # A performance that accelerates from 90 to 130 BPM: one note on every beat, plus an off-beat.
    beats = [0.5]
    for k in range(31):
        bpm = 90 + 40 * k / 30
        beats.append(beats[-1] + 60 / bpm)
    notes = [{"pitch": 60 + k % 12, "start": t, "end": t + 0.1, "velocity": 80} for k, t in enumerate(beats)]
    notes.append({"pitch": 90, "start": (beats[10] + beats[11]) / 2, "end": beats[11], "velocity": 80})
    tmap = midi.TimeMap(110, beats)
    assert tmap.is_map
    spec = [midi.MidiTrackSpec("Keys", notes)]
    path = tmp_path / "map.mid"
    midi.save_and_validate(midi.build_midi(spec, tmap), path, spec, tmap)

    import mido

    mf = mido.MidiFile(str(path))
    tempos = [m for m in mf.tracks[0] if m.type == "set_tempo"]
    assert len(tempos) > 20  # the map is written, not one average tempo
    on_ticks = sorted(t for t, m in _abs(mf.tracks[1]) if m.type == "note_on" and m.velocity and m.note != 90)
    first = on_ticks[0]
    assert all(abs(t - first - k * midi.PPQ) <= 1 for k, t in enumerate(on_ticks))  # every beat on the grid

    pm = pretty_midi.PrettyMIDI(str(path))  # independent parser that applies the tempo changes
    got = sorted((n.start, n.pitch) for n in pm.instruments[0].notes)
    want = sorted((n["start"], n["pitch"]) for n in notes)
    assert max(abs(g[0] - w[0]) for g, w in zip(got, want, strict=True)) < 0.003


def test_constant_tempo_is_the_default_and_rejects_implausible_beats():
    assert not midi.TimeMap(120).is_map
    assert not midi.TimeMap(120, [0.0, 0.05, 0.1, 0.15, 0.2]).is_map  # 1200 BPM is not a beat track


def _abs(track):
    t = 0
    for m in track:
        t += m.time
        yield t, m
