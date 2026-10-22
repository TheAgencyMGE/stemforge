"""Unit tests: security helpers, storage confinement, MIDI writer/validator, engines, analysis."""

from __future__ import annotations

import time

import mido
import numpy as np
import pytest

from stemforge.pipeline import analysis, midi, transcription
from stemforge.security import RateLimiter, safe_filename, sign, sniff_audio_format, verify_signature
from stemforge.storage import LocalStorage, StorageError, validate_key

# ---------------------------------------------------------------- security


@pytest.mark.parametrize(
    "head,fmt",
    [
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", "wav"),
        (b"fLaC\x00\x00\x00\x22" + b"\x00" * 8, "flac"),
        (b"ID3\x04\x00\x00\x00\x00\x00\x00\x00\x00", "mp3"),
        (b"\xff\xfb\x90\x64" + b"\x00" * 8, "mp3"),
        (b"OggS\x00\x02" + b"\x00" * 8, "ogg"),
        (b"FORM\x00\x00\x00\x00AIFF", "aiff"),
        (b"\x00\x00\x00\x20ftypM4A \x00\x00", "m4a"),
        (b"<html><body>", None),
        (b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00", None),  # Windows executable
        (b"PK\x03\x04" + b"\x00" * 8, None),
        (b"short", None),
    ],
)
def test_sniff(head, fmt):
    assert sniff_audio_format(head) == fmt


@pytest.mark.parametrize(
    "name,expected",
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\Windows\\system32\\evil.wav", "evil.wav"),
        ("my song (live).mp3", "my song (live).mp3"),
        ("Beyoncé – Halo.flac", "Beyonce Halo.flac"),
        ("....", "audio"),
        ("", "audio"),
        ("a" * 300 + ".wav", "a" * 120),
    ],
)
def test_safe_filename(name, expected):
    assert safe_filename(name) == expected


def test_signatures_expire_and_bind_to_artifact():
    exp = int(time.time()) + 60
    sig = sign("k", "abc", exp)
    assert verify_signature("k", "abc", exp, sig)
    assert not verify_signature("k", "abd", exp, sig)
    assert not verify_signature("other", "abc", exp, sig)
    assert not verify_signature("k", "abc", int(time.time()) - 1, sign("k", "abc", int(time.time()) - 1))


def test_rate_limiter():
    rl = RateLimiter(per_minute=3)
    assert [rl.allow("ip") for _ in range(4)] == [True, True, True, False]
    assert rl.allow("other-ip")


def test_storage_confinement(tmp_path):
    s = LocalStorage(tmp_path / "store")
    for bad in ["../x", "a/../../x", "/abs", "a//b", "", "a/"]:
        with pytest.raises(StorageError):
            validate_key(bad)
    src = tmp_path / "f.bin"
    src.write_bytes(b"hello")
    s.put_file("projects/p1/a.bin", src, "application/octet-stream")
    assert s.exists("projects/p1/a.bin")
    assert s.open("projects/p1/a.bin").read() == b"hello"
    s.delete_prefix("projects/p1")
    assert not s.exists("projects/p1/a.bin")


# ---------------------------------------------------------------- MIDI


def _note(pitch, start, end, vel=90, bends=None):
    return {"id": f"{pitch}-{start}", "pitch": pitch, "start": start, "end": end, "velocity": vel,
            "confidence": 0.9, "bends": bends}


def test_midi_roundtrip_timing_and_tracks(tmp_path):
    bpm = 100.0
    lead = [_note(60, 0.0, 0.5), _note(64, 0.6, 1.2, bends=[[0.0, 0.0], [0.2, 0.5], [0.6, 0.0]])]
    drums = [_note(36, 0.0, 0.06), _note(38, 0.6, 0.66), _note(42, 0.3, 0.36)]
    specs = [midi.MidiTrackSpec("Lead", lead, 53), midi.MidiTrackSpec("Drums", drums, 0, is_drum=True)]
    mf = midi.build_midi(specs, bpm, "4/4", offset=2.0, key={"tonic": "A", "mode": "minor"},
                         markers=[{"start": 0.0, "label": "A"}])
    path = tmp_path / "x.mid"
    report = midi.save_and_validate(mf, path, specs, bpm, offset=2.0)
    assert [t["notes"] for t in report["tracks"]] == [2, 3]

    # Independent check with pretty_midi: absolute times include the section offset.
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    by_name = {i.name: i for i in pm.instruments}
    assert by_name["Drums"].is_drum
    assert by_name["Lead"].program == 53
    starts = sorted(n.start for n in by_name["Lead"].notes)
    assert starts == pytest.approx([2.0, 2.6], abs=1e-3)
    assert pm.get_tempo_changes()[1][0] == pytest.approx(bpm, abs=0.01)
    assert len(by_name["Lead"].pitch_bends) >= 2
    # Drums must be on channel 10 (index 9)
    raw = mido.MidiFile(str(path))
    chans = {m.channel for t in raw.tracks for m in t if m.type == "note_on"}
    assert 9 in chans and len(chans) == 2


def test_midi_overlapping_same_pitch_is_unambiguous(tmp_path):
    notes = [_note(60, 0.0, 1.0), _note(60, 0.5, 1.5), _note(60, 0.5, 0.7)]  # overlap + duplicate start
    specs = [midi.MidiTrackSpec("Keys", notes)]
    mf = midi.build_midi(specs, 120, "4/4")
    report = midi.save_and_validate(mf, tmp_path / "o.mid", specs, 120)
    assert report["tracks"][0]["notes"] == 2
    import pretty_midi

    ns = sorted(pretty_midi.PrettyMIDI(str(tmp_path / "o.mid")).instruments[0].notes, key=lambda n: n.start)
    assert ns[0].end == pytest.approx(0.5, abs=1e-3)  # first note truncated at the re-attack


def test_midi_empty_and_out_of_range(tmp_path):
    specs = [midi.MidiTrackSpec("Empty", []), midi.MidiTrackSpec("Bad", [_note(200, 0, 1), _note(60, 0, 1)])]
    mf = midi.build_midi(specs, 0, "3/4")  # tempo 0 falls back to 120
    report = midi.save_and_validate(mf, tmp_path / "e.mid", specs, 0)
    assert [t["notes"] for t in report["tracks"]] == [0, 1]


# ---------------------------------------------------------------- transcription engines

SR = 44100


def _synth_melody(pitches, dur=0.4, sr=SR):
    out = []
    for p in pitches:
        t = np.arange(int(dur * sr)) / sr
        f = 440 * 2 ** ((p - 69) / 12)
        env = np.minimum(1, t / 0.01) * np.minimum(1, (dur - t) / 0.03)
        x = sum(0.5 ** k * np.sin(2 * np.pi * f * (k + 1) * t) for k in range(4)) * env * 0.3
        out.append(x)
        out.append(np.zeros(int(0.05 * sr)))
    y = np.concatenate(out)
    return np.stack([y, y]).astype(np.float32)


def test_pyin_engine_recovers_melody():
    pitches = [57, 60, 64, 62, 59, 57]
    audio = _synth_melody(pitches)
    params = transcription.resolve_params("pyin", {})
    feats = transcription.compute_features("pyin", audio, SR, params)
    notes = transcription.notes_from_features("pyin", feats, audio, SR, params)
    assert [n["pitch"] for n in notes] == pitches
    assert all(0 <= n["confidence"] <= 1 for n in notes)
    assert all(1 <= n["velocity"] <= 127 for n in notes)
    starts = [n["start"] for n in notes]
    assert starts == sorted(starts)
    assert abs(starts[1] - 0.45) < 0.05


def test_basic_pitch_engine_on_chord():
    sr = SR
    t = np.arange(int(1.5 * sr)) / sr
    chord = [60, 64, 67]
    y = sum(0.2 * np.sin(2 * np.pi * 440 * 2 ** ((p - 69) / 12) * t) * np.exp(-t * 1.5) for p in chord)
    audio = np.stack([y, y]).astype(np.float32)
    params = transcription.resolve_params("basic-pitch", {})
    feats = transcription.compute_features("basic-pitch", audio, sr, params)
    notes = transcription.notes_from_features("basic-pitch", feats, audio, sr, params)
    found = {n["pitch"] for n in notes if n["start"] < 0.2}
    assert set(chord) <= found


def test_drum_nmf_engine_separates_classes():
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
            x = sosfilt(butter(4, [1200 / (sr / 2), 6000 / (sr / 2)], btype="band", output="sos"), rng.standard_normal(n)) * np.exp(-tt * 22) * 0.6
            x += 0.4 * np.sin(2 * np.pi * 190 * tt) * np.exp(-tt * 30)
        else:
            x = sosfilt(butter(4, 7500 / (sr / 2), btype="high", output="sos"), rng.standard_normal(n)) * np.exp(-tt * 70) * 0.35
        i = int(t * sr)
        y[i:i + n] += x[: len(y) - i]

    truth = {"kick": [0.25, 1.25, 2.25, 3.25], "snare": [0.75, 1.75, 2.75], "hihat": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]}
    for k, ts in truth.items():
        for t in ts:
            hit(k, t)
    audio = np.stack([y, y]).astype(np.float32)
    params = transcription.resolve_params("drum-nmf", {})
    notes = transcription.notes_from_features("drum-nmf", transcription.compute_features("drum-nmf", audio, sr, params), audio, sr, params)
    for k, pitch in transcription.GM_DRUMS.items():
        got = sorted(n["start"] for n in notes if n["pitch"] == pitch)
        matched = sum(any(abs(g - t) < 0.03 for g in got) for t in truth[k])
        assert matched >= len(truth[k]) - 1, (k, got)
        assert len(got) <= len(truth[k]) + 1, (k, got)


def test_engine_policy_is_explained_and_params_clamped():
    for stem in ["vocals", "bass", "drums", "piano", "guitar", "other", "unknown"]:
        eng, params, reason = transcription.choose_engine(stem)
        assert eng in transcription.ENGINES and len(reason) > 20
    p = transcription.resolve_params("basic-pitch", {"onset_threshold": 5, "min_note_ms": -3, "bogus": 1,
                                                     "min_freq": 400, "max_freq": 100})
    assert p["onset_threshold"] == 0.95 and p["min_note_ms"] == 10 and "bogus" not in p
    assert p["max_freq"] is None  # inverted band is dropped rather than producing no notes


# ---------------------------------------------------------------- analysis


def test_analysis_tempo_and_key():
    sr = SR
    bpm = 120
    y = np.zeros(int(12 * sr))
    click = np.sin(2 * np.pi * 1000 * np.arange(int(0.03 * sr)) / sr) * np.exp(-np.arange(int(0.03 * sr)) / sr * 80)
    for k in range(int(12 * bpm / 60)):
        i = int(k * 60 / bpm * sr)
        y[i:i + len(click)] += click
    t = np.arange(len(y)) / sr
    for p in (60, 64, 67):  # C major triad drone
        y += 0.1 * np.sin(2 * np.pi * 440 * 2 ** ((p - 69) / 12) * t)
    res = analysis.analyze(np.stack([y, y]), sr)
    assert abs(res["tempo"] - bpm) < 2 or abs(res["tempo"] - bpm / 2) < 1 or abs(res["tempo"] - bpm * 2) < 4
    assert res["key"]["tonic"] in ("C", "A", "E")  # C major or its relative/related keys
    assert res["duration"] == pytest.approx(12.0, abs=0.01)


def test_analysis_silence():
    res = analysis.analyze(np.zeros((2, SR * 3), dtype=np.float32), SR)
    assert res["silent"] and res["tempo"] is None
