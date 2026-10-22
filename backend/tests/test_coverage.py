"""Instrument coverage, drum relabeling, note review, and chunk alignment."""

from __future__ import annotations

import numpy as np
import pytest

from stemforge.pipeline import drums, instruments, midi, refine, separation, transcription
from stemforge.pipeline.analysis import estimate_meter

SR = 44100


def test_registry_does_not_invent_separators():
    instruments.validate_registry()
    violin = instruments.get_instrument("violin")
    assert violin is not None
    assert violin.separate == "unavailable"
    assert violin.model is None
    backing = instruments.get_instrument("backing_vocal")
    assert backing is not None and backing.separate == "shared"
    assert instruments.get_instrument("tambourine").transcribe == "unavailable"
    assert instruments.get_instrument("shaker").transcribe == "unavailable"
    isolated = {i.stem for i in instruments.INSTRUMENTS.values() if i.separate == "isolated"}
    assert isolated <= {"vocals", "drums", "bass", "other", "guitar", "piano"}
    assert all(row["id"] != "adtof" for row in instruments.catalog())


def test_unknown_instrument_is_absent():
    assert instruments.get_instrument("theremin") is None
    assert instruments.stem_blurb("kit_kick").startswith("NMF mask")


def test_overlap_add_preserves_time_and_a_constant():
    n = 1000
    overlap = 50
    chunk = 400
    pieces = []
    for start in range(0, n, chunk):
        lo, hi = max(0, start - overlap), min(n, start + chunk + overlap)
        pieces.append((lo, hi, np.ones((2, hi - lo), dtype=np.float32)))
    out = separation.overlap_add(pieces, n, overlap)
    assert out.shape == (2, n)
    assert np.allclose(out, 1.0, atol=1e-5)
    # An impulse in one chunk stays on that sample. Other chunks are silent there.
    impulse_at = 250
    pieces = []
    for start in range(0, n, chunk):
        lo, hi = max(0, start - overlap), min(n, start + chunk + overlap)
        buf = np.zeros((1, hi - lo), dtype=np.float32)
        if lo <= impulse_at < hi:
            buf[0, impulse_at - lo] = 1.0
        pieces.append((lo, hi, buf))
    placed = separation.overlap_add(pieces, n, overlap)
    assert int(np.argmax(placed[0])) == impulse_at


def test_drum_core_classes_are_not_relabeled():
    """Closed hats, snares and kicks in the unit-test kit stay on GM 36/38/42."""
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(0)
    sr = SR
    y = np.zeros(int(4 * sr))

    def hit(kind, t):
        n = int(0.2 * sr)
        tt = np.arange(n) / sr
        if kind == "kick":
            x = np.sin(2 * np.pi * np.cumsum(45 + 110 * np.exp(-tt * 30)) / sr) * np.exp(-tt * 9)
        elif kind == "snare":
            x = sosfilt(butter(4, [1200 / (sr / 2), 6000 / (sr / 2)], btype="band", output="sos"), rng.standard_normal(n))
            x = x * np.exp(-tt * 22) * 0.6 + 0.4 * np.sin(2 * np.pi * 190 * tt) * np.exp(-tt * 30)
        else:
            x = sosfilt(butter(4, 7500 / (sr / 2), btype="high", output="sos"), rng.standard_normal(n)) * np.exp(-tt * 70) * 0.35
        i = int(t * sr)
        y[i:i + n] += x[: len(y) - i]

    truth = {"kick": [0.25, 1.25, 2.25, 3.25], "snare": [0.75, 1.75, 2.75], "hihat": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]}
    for kind, times in truth.items():
        for t in times:
            hit(kind, t)
    audio = np.stack([y, y]).astype(np.float32)
    params = transcription.resolve_params("drum-nmf", {})
    notes = transcription.notes_from_features(
        "drum-nmf", transcription.compute_features("drum-nmf", audio, sr, params), audio, sr, params,
    )
    assert {n["pitch"] for n in notes} <= {36, 38, 42}


def test_open_hat_and_tom_are_relabeled_when_the_spectrum_says_so():
    sr = SR
    t = np.arange(int(0.7 * sr)) / sr
    hat = np.zeros(int(sr))
    # Long bright hit with some body, the shape the crash/open rule looks at.
    from scipy.signal import butter, sosfilt

    noise = sosfilt(butter(4, 5000 / (sr / 2), btype="high", output="sos"), np.random.default_rng(2).standard_normal(len(t)))
    hat[: len(t)] = noise * np.exp(-t * 2.2) * 0.4
    audio = np.stack([hat, hat]).astype(np.float32)
    coarse = [{"id": "h", "pitch": 42, "start": 0.0, "end": 0.06, "velocity": 80, "confidence": 0.8, "bends": None}]
    refined = drums.refine_drum_notes(coarse, audio, sr, {"onset_threshold": 0.5, "min_note_ms": 40})
    assert refined[0]["pitch"] in {46, 49}
    assert refined[0].get("flags")

    tom = np.sin(2 * np.pi * 160 * t) * np.exp(-t * 4) * 0.8
    tom_audio = np.zeros((2, sr), dtype=np.float32)
    tom_audio[:, : len(tom)] = tom
    kick_like = [{"id": "k", "pitch": 36, "start": 0.0, "end": 0.06, "velocity": 90, "confidence": 0.7, "bends": None}]
    relabeled = drums.refine_drum_notes(kick_like, tom_audio, sr, {"onset_threshold": 0.5, "min_note_ms": 40})
    assert relabeled[0]["pitch"] in {45, 47, 50}


def test_kit_masks_are_aligned_and_put_energy_on_the_right_piece():
    sr = SR
    y = np.zeros(int(2 * sr), dtype=np.float32)
    tt = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * np.cumsum(50 + 80 * np.exp(-tt * 25)) / sr) * np.exp(-tt * 8)
    y[int(0.2 * sr): int(0.2 * sr) + len(kick)] += kick.astype(np.float32)
    snare_t = np.arange(int(0.15 * sr)) / sr
    snare = np.sin(2 * np.pi * 190 * snare_t) * np.exp(-snare_t * 20)
    y[int(1.0 * sr): int(1.0 * sr) + len(snare)] += (0.8 * snare).astype(np.float32)
    audio = np.stack([y, y])
    pieces = drums.piece_audio(audio, sr)
    assert set(pieces) == {"kit_kick", "kit_snare", "kit_hats", "kit_perc"}
    for piece in pieces.values():
        assert piece.shape == audio.shape
    def window_rms(buf, t):
        i = int(t * sr)
        return float(np.sqrt(np.mean(buf[0, i:i + int(0.08 * sr)] ** 2)))
    assert window_rms(pieces["kit_kick"], 0.2) > window_rms(pieces["kit_snare"], 0.2)
    assert window_rms(pieces["kit_snare"], 1.0) > window_rms(pieces["kit_kick"], 1.0)


def test_bass_octave_correction_is_reversible_and_does_not_delete():
    sr = SR
    dur = 0.6
    n = int(dur * sr)
    tt = np.arange(n) / sr
    f0 = 440 * 2 ** ((40 - 69) / 12)
    saw = 2 * ((tt * f0) % 1.0) - 1
    audio = np.stack([saw, saw]).astype(np.float32) * 0.4
    claimed = [{"id": "n1", "pitch": 52, "start": 0.02, "end": 0.5, "velocity": 90, "confidence": 0.8, "bends": None}]
    corrected, changed = refine.correct_bass_octaves(claimed, audio, sr)
    assert len(corrected) == 1
    assert changed and corrected[0]["pitch"] == 40 and corrected[0]["raw_pitch"] == 52
    # A real sine at the written pitch is left alone.
    f = 440 * 2 ** ((52 - 69) / 12)
    sine = np.sin(2 * np.pi * f * tt).astype(np.float32) * 0.4
    sine_audio = np.stack([sine, sine])
    kept = [{"id": "n2", "pitch": 52, "start": 0.02, "end": 0.5, "velocity": 80, "confidence": 0.9, "bends": None}]
    out, moved = refine.correct_bass_octaves(kept, sine_audio, sr)
    assert not moved and out[0]["pitch"] == 52


def test_review_flags_without_deleting():
    notes = [
        {"id": "low", "pitch": 10, "start": 0.0, "end": 0.2, "velocity": 80, "confidence": 0.9, "bends": None},
        {"id": "a", "pitch": 40, "start": 1.0, "end": 1.05, "velocity": 70, "confidence": 0.4, "bends": None},
        {"id": "b", "pitch": 40, "start": 1.06, "end": 1.1, "velocity": 70, "confidence": 0.4, "bends": None},
        {"id": "c", "pitch": 40, "start": 1.11, "end": 1.15, "velocity": 70, "confidence": 0.4, "bends": None},
        {"id": "d", "pitch": 40, "start": 1.16, "end": 1.2, "velocity": 70, "confidence": 0.4, "bends": None},
        {"id": "long", "pitch": 36, "start": 2.0, "end": 9.0, "velocity": 40, "confidence": 0.2, "bends": None},
    ]
    result = refine.apply("bass", "basic-pitch", notes, None, SR)
    assert len(result.notes) == len(notes)
    codes = {s["code"] for s in result.suggestions}
    assert "pitch_range" in codes
    pitch_msg = next(s["message"] for s in result.suggestions if s["code"] == "pitch_range")
    assert pitch_msg.startswith("1 note sits")
    assert "fragmented" in codes
    assert "long_sustain" in codes


def test_guitar_tab_marks_collisions_as_estimates():
    notes = [
        {"id": "e", "pitch": 40, "start": 0.0, "end": 0.4, "velocity": 80, "confidence": 0.8, "bends": None},
        {"id": "f", "pitch": 41, "start": 0.01, "end": 0.4, "velocity": 80, "confidence": 0.8, "bends": None},
    ]
    tab = refine.tablature_estimate(notes)
    assert tab["estimate"] is True
    by_pitch = {e["pitch"]: e for e in tab["events"]}
    assert by_pitch[40]["string"] == 6 and by_pitch[40]["fret"] == 0
    assert by_pitch[41]["string"] is None
    assert "not detected" in tab["disclaimer"]


def test_engine_disagreement_keeps_primary_notes():
    primary = [{"id": "p", "pitch": 40, "start": 0.0, "end": 0.2, "velocity": 80, "confidence": 0.5, "bends": None}]
    secondary = [{"id": "s", "pitch": 64, "start": 1.0, "end": 1.2, "velocity": 80, "confidence": 0.5, "bends": None}]
    found = refine.engine_disagreements(primary, secondary, "pyin")
    assert found and found[0]["note_ids"] == ["p"]
    assert refine.comparison_engine("bass", "basic-pitch") == "pyin"
    assert refine.comparison_engine("guitar", "basic-pitch") is None
    assert refine.comparison_engine("drums", "drum-nmf") is None


def test_meter_stays_undecided_when_accents_are_flat():
    env = np.ones(64)
    frames = np.arange(0, 64, 4)
    assert estimate_meter(env, frames, 120)["time_signature"] is None
    accented = np.ones(64) * 0.2
    for i in range(0, 64, 16):
        accented[i] = 1.0
    meter = estimate_meter(accented, np.arange(0, 64, 4), 120)
    assert meter["time_signature"] == "4/4"
    assert 0 <= meter["confidence"] <= 1


def test_midi_structure_rejects_a_dangling_note(tmp_path):
    import mido

    path = tmp_path / "bad.mid"
    mid = mido.MidiFile(ticks_per_beat=midi.PPQ)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="Stemforge", time=0))
    mid.tracks.append(meta)
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name="Lead", time=0))
    tr.append(mido.Message("note_on", channel=0, note=60, velocity=80, time=0))
    tr.append(mido.MetaMessage("end_of_track", time=10))
    mid.tracks.append(tr)
    mid.save(str(path))
    specs = [midi.MidiTrackSpec("Lead", [{"pitch": 60, "start": 0, "end": 0.2, "velocity": 80}])]
    with pytest.raises(midi.MidiValidationError):
        midi.validate(path, specs, 120)
